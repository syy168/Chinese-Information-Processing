import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import arxiv
import requests
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun


from enum import Enum
import traceback

class DataSource(Enum):
    ARXIV = "arxiv"
    PWC = "papers_with_code"

@dataclass
class PaperData:
    """论文数据结构"""
    title: str
    authors: List[str]
    summary: str
    pdf_url: Optional[str] = None
    published: Optional[str] = None
    github_url: Optional[str] = None
    github_info: Optional[Dict] = None
    code_info: Optional[Dict] = None
    dataset_info: Optional[Dict] = None
    metrics: Optional[Dict] = None


class DSModel(LLM):
    """DeepSeek langchain 适配器"""
    deepseek_llm: Any
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, 
            run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any, ) -> str:
        response = self.deepseek_llm.complete(prompt)
        return response.text


class ArxivAPI():
    """ArXiv (+github) 数据源"""
    dataSource = "arxiv"

    def search_papers(self, query: str, max_results: int = 10) -> List[PaperData]:
        """从ArXiv搜索论文"""
        max_results_limit = min(int(max_results), 50)
        print('最终max_results:',max_results_limit)
        #client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results_limit,
            # 按时间降序，最新论文优先
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in arxiv.Search.results(search):
            paper = PaperData(
                title=result.title,
                authors=[author.name for author in result.authors],
                summary=result.summary,
                pdf_url=result.pdf_url,
                published=result.published.isoformat() if result.published else None
            )
            papers.append(self.enrich_paper_data(paper))
        
        return papers
    
    
    def getGiLinks(self, text: str) -> List[str]:
        """提取摘要中的 GitHub 链接"""
        pattern = r'https://github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+'
        return re.findall(pattern, text)
    
    def getGitInfo(self, repo_url: str) -> Dict:
        """获取GitHub仓库信息"""
        try:
            owner, repo = repo_url.split('/')[-2:]
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                metadata = response.json()
                
                # 获取README
                readme_url = f"{repo_url}/raw/main/README.md"
                readme_response = requests.get(readme_url, timeout=10)
                readme = readme_response.text if readme_response.status_code == 200 else None
                
                return {
                    "url": repo_url,
                    "description": metadata.get("description"),
                    "language": metadata.get("language"),
                    "stars": metadata.get("stargazers_count", 0),
                    "forks": metadata.get("forks_count", 0),
                    "readme": readme
                }
        except Exception as e:
            print(f"获取GitHub信息失败: {e}")
        
        return {}
    
    def enrich_paper_data(self, paper: PaperData) -> PaperData:
        """通过GitHub信息丰富论文数据"""
        github_links = self.getGiLinks(paper.summary)
        if github_links:
            paper.github_url = github_links[0]
            paper.github_info = self.getGitInfo(paper.github_url)
        return paper
    

class PWCAPI():
    """Papers with Code 数据源"""
    dataSource = "PaperWithCode"

    BASE_URL = "https://paperswithcode.com/api/v1"
    
    def __init__(self):
        """初始化PWCAPI客户端"""
        self.session = requests.Session()
    
        
    
    def getRepo(self, paper_id: str) -> Dict:
        """获取论文相关的代码仓库信息""" 
        try:
            url = f"{self.BASE_URL}/papers/{paper_id}/repositories/"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"getRepo 失败: {response.status_code}")
        except Exception as e:
            print(f"getRepo 错误: {e}")
        return {}
    
    def getDatasets(self, paper_id: str) -> Dict:
        """获取论文使用的数据集信息""" 
        try:
            url = f"{self.BASE_URL}/papers/{paper_id}/datasets/"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"getDatasets 失败: {response.status_code}")
        except Exception as e:
            print(f"getDatasets 错误: {e}")
        return {}
    
    def getEvalRes(self, paper_id: str) -> Dict:
        """获取论文的评估结果和性能指标""" 
        try:
            url = f"{self.BASE_URL}/papers/{paper_id}/results/"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"getEvalRes 失败: {response.status_code}")
        except Exception as e:
            print(f"getEvalRes 错误: {e}")
        return {}
    
    
    def search_papers(self, query: str, max_results: int = 10) -> List[PaperData]:
        """从Papers with Code搜索论文""" 
        try:
            # 搜索论文
            search_url = f"{self.BASE_URL}/papers/"
            pages = max_results // 10 + 1
            params = {
                "q": query, 
                "items_per_page": 10, # 每页返回 10 个
                "page": pages,    # 页码
                "ordering": "-stars"  # 按星标数降序排列
            }
            
            response = self.session.get(search_url, params=params, timeout=15)
            if response.status_code != 200:
                print(f"Papers with Code API请求失败: {response.status_code} - {response.text}")
                return []
            
            data = response.json()
            papers = []
            
            for item in data.get("results", []):
                paper = PaperData(
                    title=item.get("title", ""),
                    authors=item.get("authors", ""),
                    summary=item.get("abstract", ""),
                    pdf_url=item.get("url_pdf"),
                    published=item.get("published"),
                )
                
                # 丰富数据
                paper = self.enrich_paper_data(paper, item.get("id"))
                papers.append(paper)
            
            print(f"从Papers with Code找到 {len(papers)} 篇论文")
            return papers
            
        except Exception as e:
            print(f"search_papers 搜索失败: {e}")
            traceback.print_exc()  # 打印完整的错误堆栈信息
            return []
    

    def enrich_paper_data(self, paper: PaperData, paper_id: Optional[str] = None) -> PaperData:
        """通过Papers with Code API 的一些数据丰富论文信息""" 
        try:
            if paper_id:
                # 获取代码仓库信息
                paper.code_info = self.getRepo(paper_id)
                if paper.code_info.get("results"):
                        for repo in paper.code_info["results"]:
                            if "github.com" in repo.get("url", ""):
                                paper.github_url = repo["url"]
                                break
                # 获取数据集信息
                paper.dataset_info = self.getDatasets(paper_id)
                # 获取评估结果和指标
                paper.metrics = self.getEvalRes(paper_id)
                
        except Exception as e:
            print(f"enrich_paper_data 失败: {e}")
        
        return paper
    

class DocumentProcessor:
    """文档处理器，使用 LangChain 进行分割.目前只考虑了摘要内容。"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ";", ":", "，", " ", ""]
        )

    def format_paper_content(self, paper: PaperData) -> str:
        """格式化论文内容"""
        content = f"""标题: {paper.title}
作者: {', '.join(paper.authors)}
发布时间: {paper.published or '未知'}

摘要:
{paper.summary}

PDF链接: {paper.pdf_url or '无'}"""
        
        return content
    
    def process_code_info(self, paper: PaperData) -> List[Document]:
        """处理代码信息"""

        documents = []
        # GitHub信息
        if paper.github_info:
            github_text = f"""GitHub仓库: {paper.github_info.get('url', '')}
描述: {paper.github_info.get('description', '无')}
主要语言: {paper.github_info.get('language', '未知')}
Star 数: {paper.github_info.get('stars', 0)}
Fork 数: {paper.github_info.get('forks', 0)}

README:
{paper.github_info.get('readme', '无README信息')}"""
            
            doc = Document(
                text = github_text,
                metadata={
                    "source": "github",
                    "paper_title": paper.title,
                    "repo_url": paper.github_info.get('url')
                }
            )
            documents.append(doc)
            
        # Papers with Code 代码信息
        if paper.code_info:
            code_text = f"代码实现信息:\n{json.dumps(paper.code_info, ensure_ascii=False, indent=2)}"
            doc = Document(
                text = code_text,
                metadata={
                    "source": "PWC code info",
                    "paper_title": paper.title
                }
            )
            documents.append(doc)

        if paper.dataset_info:
            dataset_text = f"数据集信息:\n{json.dumps(paper.dataset_info, ensure_ascii=False, indent=2)}"
            doc = Document(
                text = dataset_text,
                metadata={
                    "source": "PWC dataset",
                    "paper_title": paper.title
                }
            )
            documents.append(doc)
        
        if paper.metrics:
            metrics_text = f"性能指标:\n{json.dumps(paper.metrics, ensure_ascii=False, indent=2)}"
            doc = Document(
                text=metrics_text,
                metadata={
                    "source": "PWC metrics",
                    "paper_title": paper.title
                }
            )
            documents.append(doc)
        
        return documents

    
    def process_papers(self, papers: List[PaperData]) -> List[Document]:
        """处理论文数据为 LlamaIndex 文档"""
        documents = []
        
        for paper in papers:
            # 主论文文档
            paper_text = self.format_paper_content(paper)
            chunks = self.text_splitter.split_text(paper_text)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    text=chunk,
                    metadata={
                        "source": "paper",
                        "title": paper.title,
                        "authors": ", ".join(paper.authors),
                        "chunk_id": i
                    }
                )
                documents.append(doc)
            
            # GitHub/代码文档
            if paper.github_info or paper.code_info:
                code_docs = self.process_code_info(paper)
                documents.extend(code_docs)
        
        return documents
    

class PromptManager:
    """提示管理器，使用 LangChain 的提示模板"""
    
    def __init__(self):
        self.templates = {
            "paper_analysis": PromptTemplate(
                input_variables=["context", "query"],
                template="""你是一个专业的学术论文分析助手。基于以下论文内容回答问题。

论文内容:
{context}

问题: {query}

请提供详细分析，包括:
1. 直接回答问题
2. 引用具体论文内容
3. 提供论文标题作为来源

回答:"""
            ),
            
            "code_analysis": PromptTemplate(
                input_variables=["context", "query"],
                template="""你是一个代码分析专家。基于以下代码和论文信息回答问题。

相关信息:
{context}

问题: {query}

请提供:
1. 算法理论基础
2. 代码实现特点
3. 性能分析
4. 使用方法

回答:"""
            ),
            
            "comprehensive": PromptTemplate(
                input_variables=["context", "query", "history"],
                template="""基于以下信息和对话历史回答问题。

对话历史:
{history}

相关信息:
{context}

当前问题: {query}

请提供准确、详细的回答:"""
            )
        }
    
    def get_template(self, template_name: str) -> PromptTemplate:
        return self.templates.get(template_name, self.templates["comprehensive"])


class RAGSystem:
    """综合 LangChain 和 LlamaIndex 的RAG系统"""
    
    def __init__(self, data_source: DataSource = DataSource.ARXIV):
        self.data_source = data_source
        self.dataAPI = ArxivAPI() if data_source == DataSource.ARXIV else PWCAPI()

        self.document_processor = DocumentProcessor()
        self.prompt_manager = PromptManager()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            input_key="query",
            return_messages=True
        )
        
        # LlamaIndex组件
        self.index = None
        self.query_engine = None
        
        # LangChain组件
        self.llm_adapter = None
        self.chains = {}
    
    def setup_llm(self, api_key: str, model: str = "deepseek-chat"):
        """设置 LLM """
        # 配置DeepSeek
        deepseek_llm = DeepSeek(
            model=model,
            api_key=api_key,
            temperature=0.1
        )
        
        # 设置全局配置
        Settings.llm = deepseek_llm
        
        # Settings.embed_model = HuggingFaceEmbedding(model_name="/home/aistudio/BAAI-bge-small-en-v1.5")
        Settings.embed_model = HuggingFaceEmbedding(model_name="../BAAI-bge-small-en-v1.5")
        
        
        
        # 创建LangChain适配器
        self.llm_adapter = DSModel(deepseek_llm = deepseek_llm)
        
        # 创建 LangChain 处理链
        for name, template in self.prompt_manager.templates.items():
            self.chains[name] = LLMChain(
                llm=self.llm_adapter,
                prompt=template,
                memory=self.memory if name == "comprehensive" else None
            )


    def search_and_index(self, query: str, max_results: int = 10) -> List[PaperData]:
        """搜索论文并构建索引"""
        print(f"使用 {self.data_source.value} 搜索论文...")
        
        # 搜索论文
        print(query, max_results)
        papers = self.dataAPI.search_papers(query, max_results)
        print(f"找到 {len(papers)} 篇论文")
        
        # 处理文档
        documents = self.document_processor.process_papers(papers)
        print(f"处理了 {len(documents)} 个文档块")
        
        # 利用 LlamaIndex 构建索引
        self.index = VectorStoreIndex.from_documents(
            documents,
            transformations=[SentenceSplitter(chunk_size=1000, chunk_overlap=200)]
        )
        
        # 创建查询引擎
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="tree_summarize"
        )

        print("索引构建完成")
        return papers
    
    def query(self, question: str, query_type: str = "comprehensive", selected_papers=None, selected_codes=None) -> Dict[str, Any]:
        """智能查询"""
        if not self.index:
            raise ValueError("请先搜索论文并构建索引")
        
        # 使用 LlamaIndex 检索相关文档
        if selected_papers or selected_codes:
            # 如果用户选择了特定文章或代码，则只从这些内容中检索
            nodes = []
            
            # 从索引中获取所有节点
            all_nodes = self.index.docstore.docs.values()
            
            # 根据选择的文章筛选节点
            if selected_papers:
                for node in all_nodes:
                    if 'title' in node.metadata and node.metadata['title'] in selected_papers:
                        nodes.append(node)
            
            # 根据选择的代码筛选节点
            if selected_codes and not nodes:  # 如果没有选择文章或没有找到匹配的文章节点
                for node in all_nodes:
                    if 'source' in node.metadata and node.metadata['source'] in ['github', 'PWC code info'] and \
                       'paper_title' in node.metadata and node.metadata['paper_title'] in selected_codes:
                        nodes.append(node)
            
            # 如果没有找到匹配的节点，则使用默认检索方法
            if not nodes:
                retriever = VectorIndexRetriever(index=self.index, similarity_top_k=5)
                nodes = retriever.retrieve(question)
        else:
            # 默认检索方法
            retriever = VectorIndexRetriever(index=self.index, similarity_top_k=5)
            nodes = retriever.retrieve(question)
        
        # 准备上下文
        context = "\n\n".join([node.text for node in nodes])
        
        # 获取对话历史
        history_dict = self.memory.load_memory_variables({})
        history = history_dict.get("chat_history", "")

        # 使用 LangChain 链处理
        if query_type in self.chains:
            chain = self.chains[query_type]
            if query_type == "comprehensive":
                result = chain.invoke({"context": context, "query": question, "history": history})
            else:
                result = chain.invoke({"context": context, "query": question})
            #response = str(result)
            if hasattr(result, 'text'):
                response = result.text
            elif isinstance(result, dict) and 'text' in result:
                response = result['text']
            else:
                response = str(result)
        else:
            # 回退到 LlamaIndex
            response = str(self.query_engine.query(question))
        

        return {
            "query": question,
            "response": response,
            "sources": [{
                "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                "metadata": node.metadata
            } for node in nodes],
            "method": f"langchain_{query_type}" if query_type in self.chains else "llamaindex"
        }
    

    def clear_memory(self):
        """清除对话记忆"""
        self.memory.clear()
    
    def save_data(self, papers: List[PaperData], filename: str):
        """保存论文数据"""
        data = []
        for paper in papers:
            paper_dict = {
                "title": paper.title,
                "authors": paper.authors,
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
                "published": paper.published,
                "github_url": paper.github_url,
                "github_info": paper.github_info,
                "code_info": paper.code_info,
                "dataset_info": paper.dataset_info,
                "metrics": paper.metrics
            }
            data.append(paper_dict)
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"数据已保存到 {filename}")
    
    def load_data(self, filename: str) -> List[PaperData]:
        """加载论文数据"""
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        papers = []
        for item in data:
            paper = PaperData(
                title=item["title"],
                authors=item["authors"],
                summary=item["summary"],
                pdf_url=item.get("pdf_url"),
                published=item.get("published"),
                github_url=item.get("github_url"),
                github_info=item.get("github_info"),
                code_info=item.get("code_info"),
                dataset_info=item.get("dataset_info"),
                metrics=item.get("metrics")
            )
            papers.append(paper)
        
        return papers
# 在RAGSystem类中添加以下方法

    def add_paper_from_arxiv_link(self, arxiv_link: str) -> Optional[PaperData]:
        """从ArXiv链接添加单篇论文"""
        try:
            # 从链接中提取论文ID
            import re
            arxiv_id = re.search(r'arxiv\.org/(?:abs|pdf)/([\d\.]+)(?:v\d+)?', arxiv_link)
            # print(arxiv_id)
            if not arxiv_id:
                return None
            
            arxiv_id = arxiv_id.group(1)
            
            # 使用arxiv API获取论文信息
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            
            for result in client.results(search):
                paper = PaperData(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    summary=result.summary,
                    pdf_url=result.pdf_url,
                    published=result.published.isoformat() if result.published else None
                )
                
                # 丰富论文数据
                if self.data_source == DataSource.ARXIV:
                    paper = self.dataAPI.enrich_paper_data(paper)
                else:
                    # 如果当前数据源是PWC，但我们添加的是ArXiv论文，仍然使用ArXiv API丰富数据
                    arxiv_api = ArxivAPI()
                    paper = arxiv_api.enrich_paper_data(paper)
                
                return paper
            
            return None
        except Exception as e:
            print(f"从ArXiv链接添加论文失败: {e}")
            return None
    
    def optimize_query(self, query: str) -> str: 
        """使用 DeepSeek 优化搜索关键词，并将其转换为 arXiv 支持的布尔查询语法"""
        if not self.llm_adapter:
            raise ValueError("请先配置LLM")
        
        # 构建提示模板
        prompt_template = """
        你是一个专业的对于arXiv论文搜索专家。请将以下搜索关键词优化为更专业、更精确的学术搜索词，以便在学术论文数据库中检索相关论文。

        原始搜索关键词: {query}

        请考虑以下几点：
        1. 使用该领域的专业术语和标准表达方式
        2. 使用恰当的布尔表达式EXP和字段type连接关键词，EXP可选AND OR,type可选为ti（tiltle）abs（abstract）all（所有），
        3. 保持简洁，通常不超过3个关键词组合
        4. 保持英文表达
        5. 如果原始关键词已经是专业的学术搜索词，可以保持不变或做微小调整
        6. 返回类似的字符串type1:keyword1 EXP type2:keyword3 EXP type3:keyword3

        优化后的搜索字符串（不要包含任何解释）:
        """

        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain

        optimize_prompt = PromptTemplate(
            input_variables=["query"],
            template=prompt_template
        )

        optimize_chain = LLMChain(
            llm=self.llm_adapter,
            prompt=optimize_prompt
        )

        # 调用 LLM 优化查询
        result = optimize_chain.invoke({"query": query})

        # 提取结果
        if hasattr(result, 'text'):
            optimized_query = result.text.strip()
        elif isinstance(result, dict) and 'text' in result:
            optimized_query = result['text'].strip()
        else:
            optimized_query = str(result).strip()

        # 如果为空，直接返回原始
        if not optimized_query:
            return query
        return optimized_query

        #  将 "xxx", "yyy" 格式转换为 arXiv 支持的查询格式
        # import re
        # keywords = re.findall(r'"([^"]+)"', optimized_query)
        # if not keywords:
        #     return f'all:"{optimized_query}"'
        #
        # # 推荐方式：拼接为空格，单一 all 查询
        # arxiv_query = 'OR'.join(f'"{kw}"' for kw in keywords)
        # return arxiv_query


if __name__ == "__main__":

    # 配置API密钥
    api_key = "sk-a89c7976401d431d9bc25a556c2fb812"  # 请替换为实际密钥
    
    print("=== Link Start ===")
    
    # 演示ArXiv + GitHub数据源
    print("\n1. 使用ArXiv + GitHub数据源")
    rag_arxiv = RAGSystem(DataSource.ARXIV)
    rag_arxiv.setup_llm(api_key)
    
    papers_arxiv = rag_arxiv.search_and_index("semantic segmentation", max_results=5)
    rag_arxiv.save_data(papers_arxiv, "papers_arxiv.json")
    
    # 查询示例
    result = rag_arxiv.query("请总结主要的语义分割算法", "paper_analysis")
    print(f"查询结果: {result['response']}...")
    
    # 演示 Papers with Code数据源
    print("\n2. 使用Papers with Code数据源")
    rag_pwc = RAGSystem(DataSource.PWC)
    rag_pwc.setup_llm(api_key)
    
    papers_pwc = rag_pwc.search_and_index("semantic segmentation", max_results=10)
    rag_pwc.save_data(papers_pwc, "papers_pwc.json")
    
    # 代码相关查询
    result = rag_pwc.query("这些算法的代码实现有什么特点？", "code_analysis")
    print(f"代码分析结果: {result['response']}...")
    
    # 对话功能演示
    print("\n3. 对话功能演示")
    result1 = rag_arxiv.query("哪个算法性能最好？", "comprehensive")
    result2 = rag_arxiv.query("它在哪些数据集上测试过？", "comprehensive")
    print(type(result2['response']))
    print(f"对话结果: {result2['response']}...")
    
    print("\n演示完成！")
