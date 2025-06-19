import streamlit as st
import os
import json
from datetime import datetime
import streamlit as st
from rag import RAGSystem, DataSource, PaperData
# 在文件顶部导入部分添加
import re


import re
import base64
import requests
import io
import streamlit.components.v1 as components

def render_markdown_with_mermaid(text: str):
    pattern = r'```mermaid([\s\S]*?)```'
    matches = re.finditer(pattern, text)

    last_end = 0
    for i, match in enumerate(matches):
        start, end = match.span()
        code = match.group(1).strip()
        chart_id = f"chart_{i}"
        code_json = json.dumps(code)

        st.markdown(f"### Mermaid 图 #{i + 1}")
        show_rendered = st.checkbox(f"切换为图形化显示 #{i + 1}", value=False, key=f"toggle_{i}")

        if not show_rendered:
            st.code(code, language="mermaid")
        else:
            html = f"""
            <html>
            <head>
              <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
              <style>
                .download-btn {{
                    position: absolute;
                    top: 5px;
                    right: 5px;
                    background: rgba(0,0,0,0.5);
                    border: none;
                    width: 28px;
                    height: 28px;
                    border-radius: 4px;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    opacity: 0.7;
                    transition: opacity 0.2s;
                    padding: 0;
                }}
                .download-btn:hover {{
                    opacity: 1;
                }}
                .download-btn svg {{
                    fill: white;
                    width: 18px;
                    height: 18px;
                }}
              </style>
            </head>
            <body>
              <div id="{chart_id}_container" style="position: relative;">
                <div class="mermaid" id="{chart_id}">{code}</div>
                <button class="download-btn" onclick="downloadSVG()" title="下载 SVG">
                  <svg viewBox="0 0 24 24">
                    <path d="M5 20h14v-2H5v2zm7-18L5.33 9.67h3.84v6.66h5.66v-6.66h3.84L12 2z"/>
                  </svg>
                </button>
              </div>

              <script>
                const code = {code_json};
                mermaid.initialize({{ startOnLoad: false }});

                setTimeout(() => {{
                    const el = document.getElementById("{chart_id}");
                    el.innerHTML = code;
                    mermaid.init(undefined, "#" + el.id);
                }}, 100);

                function downloadSVG() {{
                    const svgEl = document.querySelector("#{chart_id} svg");
                    if (!svgEl) {{
                        alert("SVG 尚未渲染，请稍候");
                        return;
                    }}
                    const serializer = new XMLSerializer();
                    const svgContent = serializer.serializeToString(svgEl);
                    const blob = new Blob([svgContent], {{ type: "image/svg+xml" }});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = "{chart_id}.svg";
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }}
              </script>
            </body>
            </html>
            """
            components.html(html, height=500, scrolling=True)

    # 显示尾部 markdown
    tail = text[end:]
    if tail.strip():
        st.markdown(tail)



# 页面配置
st.set_page_config(
    page_title="RAG 论文检索系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题
st.title("📚 RAG 论文检索与问答系统")
st.markdown("基于 LangChain + LlamaIndex 的智能论文检索和代码分析系统")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # 数据源选择
    st.subheader("📊 数据源配置")
    data_source = st.selectbox(
        "选择数据源",
        ["ArXiv", "Papers with Code"],
        help="选择论文数据来源"
    )
    
    # API 配置
    st.subheader(" DeepSeek API 配置")
    api_key = st.text_input(
        "API Key", 
        type="password", 
        value="sk-a89c7976401d431d9bc25a556c2fb812",
        help="请输入您的 DeepSeek API 密钥"
    )
    
    model_name = st.selectbox(
        "模型选择",
        ["deepseek-chat", "deepseek-coder"],
        help="选择要使用的 DeepSeek 模型"
    )

# 初始化会话状态
if 'papers' not in st.session_state:
    st.session_state.papers = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False
if 'current_data_source' not in st.session_state:
    st.session_state.current_data_source = None

# 配置 API 和 RAG 系统
def configure_rag_system():
    if api_key:
        # 确定数据源
        selected_data_source = DataSource.ARXIV if data_source == "ArXiv" else DataSource.PWC
        
        # 检查是否需要重新创建RAG系统
        if (st.session_state.rag_system is None or 
            st.session_state.current_data_source != selected_data_source):
            
            # 创建新的RAG系统
            st.session_state.rag_system = RAGSystem(selected_data_source)
            st.session_state.current_data_source = selected_data_source
        
        # 设置LLM
        st.session_state.rag_system.setup_llm(api_key, model_name)
        st.session_state.api_configured = True
        return True
    return False

# 主界面
#tab1, tab2, tab3 = st.tabs(["📖 论文搜索", "🤖 智能问答", "📊 数据管理"])
tab1, tab2, tab3, tab4 = st.tabs(["📖 论文搜索", "🤖 智能问答", "📊 数据管理", "🎨 图表渲染"])

with tab1:
    st.header(f"📖 论文搜索 - {data_source}")
    
    col1, col2 = st.columns([3, 1])
    
    # 在搜索关键词输入框下方添加提示词优化选项
    with col1:
        search_keyword = st.text_input(
            "搜索关键词",
            placeholder="例如: semantic segmentation, transformer, computer vision",
            help="输入您想搜索的论文关键词"
        )
        use_query_optimization = st.checkbox(
            "使用提示词优化", 
            value=False,
            help="使用大模型对搜索关键词进行优化，生成更专业的学术搜索词"
        )
    
    with col2:
        max_results = st.number_input(
            "最大结果数",
            min_value=1,
            max_value=50,
            value=10,
            help="限制搜索结果数量"
        )
    
    # 显示数据源信息
    if data_source == "ArXiv":
        st.info("🔍 将从 ArXiv 搜索论文并自动提取 GitHub 代码仓库信息")
    else:
        st.info("🔍 将从 Papers with Code 搜索论文，包含代码实现、数据集和性能指标")
    
    if st.button("🔍 开始搜索", type="primary"):
        if not search_keyword:
            st.error("请输入搜索关键词")
        elif not configure_rag_system():
            st.error("请先配置 DeepSeek API")
        else:
            # 如果选择了提示词优化，先优化搜索关键词
            if use_query_optimization:
                with st.spinner("正在优化搜索关键词..."):
                    optimize_keyword = st.session_state.rag_system.optimize_query(search_keyword)
                    # 显示优化后的关键词
                    st.info(f"优化后的搜索关键词: {optimize_keyword}")
                    # 使用优化后的关键词搜索
                    # papers = st.session_state.rag_system.search_and_index(optimized_keyword, max_results)
                    
            with st.spinner(f"正在使用 {data_source} 搜索论文..."):
                try:                   
                    if use_query_optimization:
                        papers = st.session_state.rag_system.search_and_index(optimize_keyword, max_results)
                    else:
                        papers = st.session_state.rag_system.search_and_index(search_keyword, max_results)
                    if papers is None :
                        st.info("未搜索到结果，请重新搜索")
                    else:
                        st.session_state.papers = papers

                        # 保存数据
                        filename = f"papers_{data_source.lower().replace(' ', '_')}.json"
                        st.session_state.rag_system.save_data(papers, filename)

                        st.success(f"✅ 成功处理 {len(papers)} 篇论文并构建索引")

                        # 显示统计信息
                        github_count = sum(1 for p in papers if p.github_info or p.code_info)
                        dataset_count = sum(1 for p in papers if p.dataset_info)
                        metrics_count = sum(1 for p in papers if p.metrics)

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📄 论文总数", len(papers))
                        with col2:
                            st.metric("💻 包含代码", github_count)
                        with col3:
                            st.metric("📊 包含数据集", dataset_count)
                        with col4:
                            st.metric("📈 包含指标", metrics_count)
                        
                except Exception as e:
                    st.error(f"搜索过程中出现错误: {str(e)}")
                    st.exception(e)
    
    # 显示搜索结果
    if st.session_state.papers:
        st.subheader("📋 搜索结果")
        
        for i, paper in enumerate(st.session_state.papers):
            with st.expander(f"📄 {paper.title[:100]}..."):
                st.write(f"**作者:** {', '.join(paper.authors)}")
                st.write(f"**发布时间:** {paper.published or '未知'}")
                st.write(f"**摘要:** {paper.summary[:500]}...")
                if paper.pdf_url:
                    st.write(f"**PDF链接:** [下载PDF]({paper.pdf_url})")
                
                # 显示 GitHub 信息
                if paper.github_info:
                    github_info = paper.github_info
                    st.write(f"**🔗 GitHub仓库:** [查看代码]({github_info.get('url', '')})")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⭐ Stars", github_info.get('stars', 0))
                    with col2:
                        st.metric("🍴 Forks", github_info.get('forks', 0))
                    with col3:
                        st.write(f"**语言:** {github_info.get('language', '未知')}")
                
                # 显示 Papers with Code 信息
                if paper.code_info:
                    st.write("**💻 代码实现:**")
                    if isinstance(paper.code_info, dict) and paper.code_info.get('results'):
                        for repo in paper.code_info['results'][:3]:  # 显示前3个代码仓库
                            st.write(f"- [{repo.get('name', '未知')}]({repo.get('url', '#')})")
                
                if paper.dataset_info:
                    st.write("**📊 相关数据集:**")
                    if isinstance(paper.dataset_info, dict) and paper.dataset_info.get('results'):
                        for dataset in paper.dataset_info['results'][:3]:  # 显示前3个数据集
                            st.write(f"- {dataset.get('name', '未知')}")
                
                if paper.metrics:
                    st.write("**📈 性能指标:**")
                    if isinstance(paper.metrics, dict) and paper.metrics.get('results'):
                        for metric in paper.metrics['results'][:3]:  # 显示前3个指标
                            task = metric.get('task', {}).get('name', '未知任务')
                            dataset = metric.get('dataset', {}).get('name', '未知数据集')
                            st.write(f"- {task} on {dataset}")
                            # if metric.get('metrics'):
                            #     for m in metric['metrics'][:2]:  # 显示前2个指标值,有bug
                            #         st.write(f"  - {m.get('name', '未知指标')}: {m.get('value', 'N/A')}")

with tab2:
    st.header("🤖 智能问答")
    
    if not st.session_state.rag_system:
        st.warning("⚠️ 请先配置API并搜索论文以构建知识库")
    else:
        # 查询类型选择
        st.subheader("🔍 查询类型")
        query_type = st.selectbox(
            "选择查询类型",
            ["comprehensive", "paper_analysis", "code_analysis"],
            help="选择不同的查询模式"
        )
        
        # 添加选择特定文章或代码的功能
        selected_papers = None
        selected_codes = None
        
        if query_type == "paper_analysis":
            # 如果选择了论文分析模式，显示论文选择框
            if st.session_state.papers:
                paper_titles = [paper.title for paper in st.session_state.papers]
                selected_papers = st.multiselect(
                    "选择要分析的论文",
                    paper_titles,
                    help="选择特定的论文进行分析，不选择则分析所有相关论文"
                )
        
        elif query_type == "code_analysis":
            # 如果选择了代码分析模式，显示代码选择框
            if st.session_state.papers:
                # 筛选有代码的论文
                papers_with_code = [paper.title for paper in st.session_state.papers 
                                   if paper.github_info or paper.code_info]
                if papers_with_code:
                    selected_codes = st.multiselect(
                        "选择要分析的代码",
                        papers_with_code,
                        help="选择特定论文的代码进行分析，不选择则分析所有相关代码"
                    )
                else:
                    st.info("没有找到包含代码的论文")
        
        # 预设问题
        st.subheader("💡 预设问题")
        preset_questions = []
        if query_type == "comprehensive":
            preset_questions=[
                "这些论文中有哪些主要的算法创新？",
                "请总结这些研究的核心贡献",
                "有哪些代码实现可以参考？",
                "这些方法的性能如何？"
            ]
        elif query_type == "paper_analysis":
            preset_questions = [
                "请总结论文的创新点？",
                "请总结论文的核心贡献",
            ]
        elif query_type == "code_analysis":
            preset_questions = [
                "请总结代码的创新点？",
                "请总结代码的流程，用mermaid绘制",
            ]
        
        for question in preset_questions:
            if st.button(question, key=f"preset_{question}"):
                with st.spinner("🔍 正在分析..."):
                    try:
                        result = st.session_state.rag_system.query(
                            question, 
                            query_type,
                            selected_papers=selected_papers,
                            selected_codes=selected_codes
                        )
                        st.success("✅ 分析完成")
                        st.write("**回答:**")
                        # 处理回答中的Mermaid图表
                        response_text = result['response']
                        # 查找Mermaid代码块
                        mermaid_blocks = re.findall(r'```mermaid\n([\s\S]*?)\n```', response_text)

                        # 如果找到Mermaid代码块，替换并渲染
                        if mermaid_blocks:
                            # 分割文本
                            parts = re.split(r'```mermaid\n[\s\S]*?\n```', response_text)

                            # 交替显示文本和Mermaid图表
                            for i in range(len(parts)):
                                if parts[i].strip():
                                    st.markdown(parts[i])
                                if i < len(mermaid_blocks):
                                    default_code = mermaid_blocks[i]
                                    st.code(default_code, language='mermaid')  # 显示 Mermaid 源码
                                    # user_code = st.text_area(f"编辑 Mermaid 代码块 ", value=default_code, height=300)
                                    safe_code = default_code.replace("\\", "\\\\").replace("`", "\\`").replace("\n",
                                                                                                               "\\n")

                                    # 添加按钮来控制渲染,每次点击按钮会重新运行脚本，回答会被覆盖，智能回答不含修改后再渲染功能
                                    # if st.button(f"渲染图表"):
                                    # 构建 Mermaid HTML
                                    with st.expander("查看渲染后的图"):
                                        html_code = f"""
                                        <div id="mermaid-container">
                                          <div class="mermaid">
                                          {default_code}
                                          </div>
                                        </div>

                                        <div id="error-message" style="color:red; font-weight:bold;"></div>

                                        <script type="module">
                                          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';

                                          const code = `{safe_code}`;

                                          try {{
                                              mermaid.parse(code);  // 检查语法
                                              mermaid.initialize({{ startOnLoad: true }});
                                          }} catch (e) {{
                                              const container = document.getElementById("mermaid-container");
                                              const errorDiv = document.getElementById("error-message");
                                              container.innerHTML = "";  // 清空图形区域
                                              errorDiv.innerText = "❌ Mermaid 图语法错误: " + e.message;
                                          }}
                                        </script>
                                        """
                                        components.html(html_code, height=600, scrolling=True)
                        else:
                            # 如果没有Mermaid代码块，直接显示文本
                            st.markdown(response_text)
                        
                        # 显示来源
                        with st.expander("📚 参考来源"):
                            shown_titles=set()
                            for i, source in enumerate(result['sources']):
                                title = source['metadata'].get('title', '无标题')
                                if title not in shown_titles:
                                    st.write(f"**来源 {i+1}:**")
                                    # st.write(source['text'])
                                    # st.write(f"**元数据:** {source['metadata']}")
                                    st.write(title)
                                    st.write("---")
                                    shown_titles.add(title)
                    except Exception as e:
                        st.error(f"❌ 查询失败: {str(e)}")
        
        # 自定义问题
        st.subheader("❓ 自定义问题")
        user_question = st.text_input("请输入您的问题:")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔍 提问") and user_question:
                with st.spinner("🔍 正在分析..."):
                    try:
                        result = st.session_state.rag_system.query(
                            user_question, 
                            query_type,
                            selected_papers=selected_papers,
                            selected_codes=selected_codes
                        )
                        # 在tab2的回答显示部分添加Mermaid图表渲染功能
                        # 修改预设问题部分的回答显示
                        st.success("✅ 分析完成")
                        st.write("**回答:**")
                        # 处理回答中的Mermaid图表
                        response_text = result['response']
                        # 查找Mermaid代码块
                        mermaid_blocks = re.findall(r'```mermaid\n([\s\S]*?)\n```', response_text)
                        
                        # 如果找到Mermaid代码块，替换并渲染
                        if mermaid_blocks:
                            # 分割文本
                            parts = re.split(r'```mermaid\n[\s\S]*?\n```', response_text)
                            
                            # 交替显示文本和Mermaid图表
                            for i in range(len(parts)):
                                if parts[i].strip():
                                    st.markdown(parts[i])
                                if i < len(mermaid_blocks):
                                    default_code = mermaid_blocks[i]
                                    st.code(default_code, language='mermaid')  # 显示 Mermaid 源码
                                    # user_code = st.text_area(f"编辑 Mermaid 代码块 ", value=default_code, height=300)
                                    safe_code = default_code.replace("\\", "\\\\").replace("`", "\\`").replace("\n","\\n")

                                    # 添加按钮来控制渲染,每次点击按钮会重新运行脚本，回答会被覆盖，智能回答不含修改后再渲染功能
                                    # if st.button(f"渲染图表"):
                                    # 构建 Mermaid HTML
                                    with st.expander("查看渲染后的图"):
                                        html_code = f"""
                                            <div id="mermaid-container">
                                              <div class="mermaid">
                                              {default_code}
                                              </div>
                                            </div>

                                            <div id="error-message" style="color:red; font-weight:bold;"></div>

                                            <script type="module">
                                              import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';

                                              const code = `{safe_code}`;

                                              try {{
                                                  mermaid.parse(code);  // 检查语法
                                                  mermaid.initialize({{ startOnLoad: true }});
                                              }} catch (e) {{
                                                  const container = document.getElementById("mermaid-container");
                                                  const errorDiv = document.getElementById("error-message");
                                                  container.innerHTML = "";  // 清空图形区域
                                                  errorDiv.innerText = "❌ Mermaid 图语法错误: " + e.message;
                                              }}
                                            </script>
                                            """
                                        components.html(html_code, height=600, scrolling=True)
                        else:
                            # 如果没有Mermaid代码块，直接显示文本
                            st.markdown(response_text)
                                        
                        # 显示来源
                        with st.expander("📚 参考来源"):
                            shown_titles = set()
                            for i, source in enumerate(result['sources']):
                                title = source['metadata'].get('title', '无标题')
                                if title not in shown_titles:
                                    st.write(f"**来源 {i+1}:**")
                                    # st.write(source['text'])
                                    # st.write(f"**元数据:** {source['metadata']}")
                                    st.write(title)
                                    st.write("---")
                                    shown_titles.add(title)
                    except Exception as e:
                        st.error(f"❌ 查询失败: {str(e)}")
        
        with col2:
            if st.button("🗑️ 清空对话"):
                if hasattr(st.session_state.rag_system, 'memory'):
                    st.session_state.rag_system.clear_memory()
                    st.success("对话历史已清空")
        
        # 显示系统状态
        if st.session_state.rag_system:
            st.subheader("📊 系统状态")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 论文数量", len(st.session_state.papers) if st.session_state.papers else 0)
            with col2:
                st.metric("🔍 数据源", str(st.session_state.current_data_source.value) if st.session_state.current_data_source else "未选择")
            with col3:
                index_status = "✅ 已构建" if st.session_state.rag_system.index else "❌ 未构建"
                st.metric("📚 索引状态", index_status)

with tab3:
    st.header("📊 数据管理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💾 数据保存")
        if st.session_state.papers:
            st.success(f"当前已加载 {len(st.session_state.papers)} 篇论文")
            
            if st.button("📥 导出数据"):
                # 将PaperData对象转换为字典格式保存
                papers_dict = []
                for paper in st.session_state.papers:
                    paper_dict = {
                        'title': paper.title,
                        'authors': paper.authors,
                        'summary': paper.summary,
                        'published': paper.published,
                        'pdf_url': paper.pdf_url,
                        'github_url': paper.github_url,
                        'github_info': paper.github_info,
                        'code_info': paper.code_info,
                        'dataset_info': paper.dataset_info,
                        'metrics': paper.metrics
                    }
                    papers_dict.append(paper_dict)
                
                # 创建下载链接
                json_str = json.dumps(papers_dict, ensure_ascii=False, indent=2)
                st.download_button(
                    label="下载 JSON 文件",
                    data=json_str,
                    file_name=f"papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info("暂无数据")
    
    with col2:
        st.subheader("📂 数据加载")
        uploaded_file = st.file_uploader(
            "上传论文数据文件",
            type=['json'],
            help="上传之前保存的论文数据JSON文件"
        )
        
        if uploaded_file is not None:
            try:
                papers_data = json.load(uploaded_file)
                
                # 将字典转换为PaperData对象
                papers_objects = []
                for paper_dict in papers_data:
                    paper = PaperData(
                        title=paper_dict.get('title', ''),
                        authors=paper_dict.get('authors', []),
                        summary=paper_dict.get('summary', ''),
                        published=paper_dict.get('published'),
                        pdf_url=paper_dict.get('pdf_url'),
                        github_url=paper_dict.get('github_url'),
                        github_info=paper_dict.get('github_info'),
                        code_info=paper_dict.get('code_info'),
                        dataset_info=paper_dict.get('dataset_info'),
                        metrics=paper_dict.get('metrics')
                    )
                    papers_objects.append(paper)
                
                st.session_state.papers = papers_objects
                st.success(f"成功加载 {len(papers_objects)} 篇论文")
                
                if st.button("🔄 重建索引"):
                    if configure_rag_system():
                        with st.spinner("正在重建索引..."):
                            try:
                                # 处理文档并重建索引
                                documents = st.session_state.rag_system.document_processor.process_papers(papers_objects)
                                from llama_index.core import VectorStoreIndex
                                from llama_index.core.node_parser import SentenceSplitter
                                st.session_state.rag_system.index = VectorStoreIndex.from_documents(
                                    documents,
                                    transformations=[SentenceSplitter(chunk_size=1000, chunk_overlap=200)]
                                )
                                st.success("索引重建完成！")
                                st.rerun()# 强制刷新session_state
                                #st.session_state.rag_system = st.session_state.rag_system  # 强制刷新session_state
                                # if not st.session_state.rag_system:
                                #     print("none")
                            except Exception as e:
                                st.error(f"索引重建失败: {str(e)}")
                    else:
                        st.error("请先配置 API")
            except Exception as e:
                st.error(f"加载文件失败: {str(e)}")
    # 在数据管理tab中添加论文链接输入功能
    # 在数据管理tab的现有功能之后添加
    st.subheader("📝 添加单篇论文")
    paper_link = st.text_input("输入论文链接 (ArXiv或Papers with Code):", 
                              help="例如: https://arxiv.org/abs/2103.14030 或 https://paperswithcode.com/paper/...")
    
    link_source = st.radio("链接来源", ["ArXiv"], horizontal=True)
    
    if st.button("添加论文"):
        print(link_source)
        if paper_link:
            if configure_rag_system():
                with st.spinner("正在获取论文信息..."):
                    try:
                        # 根据链接来源选择不同的处理方法
                        if link_source == "ArXiv":
                            new_paper = st.session_state.rag_system.add_paper_from_arxiv_link(paper_link)
                        # else:  # Papers with Code
                        #     new_paper = st.session_state.rag_system.add_paper_from_pwc_link(paper_link)

                        if new_paper:
                            existing_titles = [paper.title for paper in st.session_state.papers]
                            if new_paper.title  in existing_titles:
                                st.error("论文已在系统中")
                            else:
                                st.session_state.papers.append(new_paper)
                                # 更新索引
                                documents = st.session_state.rag_system.document_processor.process_papers([new_paper])
                                from llama_index.core import VectorStoreIndex
                                from llama_index.core.node_parser import SentenceSplitter

                                # 如果索引已存在，添加到现有索引
                                if st.session_state.rag_system.index:
                                    st.session_state.rag_system.index.insert_nodes(documents)
                                else:  # 否则创建新索引
                                    st.session_state.rag_system.index = VectorStoreIndex.from_documents(
                                        documents,
                                        transformations=[SentenceSplitter(chunk_size=1000, chunk_overlap=200)]
                                    )

                                st.success(f"成功添加论文: {new_paper.title}")
                                st.rerun()  # 刷新页面显示新添加的论文
                        else:
                            st.error("无法获取论文信息，请检查链接是否正确")
                    except Exception as e:
                        st.error(f"无法获取论文信息，请检查链接是否正确: {str(e)}")
            else:
                st.error("请先配置 API")
        else:
            st.warning("请输入论文链接")
    # 系统状态
    st.subheader("🔧 系统状态")
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    with status_col1:
        api_status = "✅ 已配置" if st.session_state.api_configured else "❌ 未配置"
        st.metric("API 状态", api_status)
    
    with status_col2:
        papers_count = len(st.session_state.papers) if st.session_state.papers else 0
        st.metric("论文数量", papers_count)
    
    with status_col3:
        github_count = sum(1 for paper in st.session_state.papers if paper.github_info) if st.session_state.papers else 0
        st.metric("GitHub链接", github_count)
    
    with status_col4:
        rag_status = "✅ 已构建" if st.session_state.rag_system else "❌ 未构建"
        st.metric("RAG系统", rag_status)
    
    # 显示论文列表
    if st.session_state.papers:
        with st.expander("📋 论文列表"):
            for i, paper in enumerate(st.session_state.papers):
                authors_str = ', '.join(paper.authors[:3])
                if len(paper.authors) > 3:
                    authors_str += f" 等{len(paper.authors)}人"
                st.write(f"{i+1}. **{paper.title}** - {authors_str}")
                if paper.github_info:
                    st.write(f"   🔗 GitHub: {paper.github_info.get('url', 'N/A')}")
                if paper.code_info:
                    st.write(f"   💻 代码实现: 可用")

with tab4:
    if not st.session_state.rag_system:
        st.warning("⚠️ 请先配置API并搜索论文以构建知识库")
    else:
        st.header("🎨 图表渲染")

        # 全局引入 Mermaid.js 并初始化
        st.markdown("""
        <script>
            if (!window.mermaidLoaded) {
                window.mermaidLoaded = true;
                var script = document.createElement('script');
                script.src = "https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js";
                script.onload = function () {
                    mermaid.initialize({
                        startOnLoad: false,
                        theme: 'base',
                        securityLevel: 'loose'
                    });
                    console.log("Mermaid 已加载并初始化");
                };
                document.head.appendChild(script);
            }
        </script>
        """, unsafe_allow_html=True)
        
        gannt_prompt = '''请根据论文内容，按论文时间或算法提出时间顺序（以年月日为单位）生成其核心算法演进图，以Mermaid Gantt 图表形式输出'''

        result = st.session_state.rag_system.query(gannt_prompt, query_type = "comprehensive")
        st.success("✅ 生成完成")
        st.write("**算法演进图:**")
        model_output = result['response']
        print(model_output)
        
        with st.spinner(f"正在生成. Mermaid 渲染..."):
            render_markdown_with_mermaid(model_output)
        # 保存原始输出
        st.session_state.model_output = model_output
        st.success("✅ 内容渲染完成！")



        # 下载功能
        if hasattr(st.session_state, 'model_output'):
            st.download_button(
                label="下载原始内容",
                data=st.session_state.model_output,
                file_name="model_output.txt",
                mime="text/plain"
            )


# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>基于 LangChain + LlamaIndex 和 Streamlit 构建的智能论文检索系统</p>
        <p>支持 ArXiv/Papers with Code 论文搜索、GitHub 代码分析和智能问答</p>
    </div>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    # 使用 Streamlit 内部 API 启动
    try:
        import streamlit.web.cli as stcli
        import sys
        
        # 设置命令行参数
        sys.argv = ["streamlit", "run", __file__]
        stcli.main()
    except Exception as e:
        print(f"启动失败：{e}")
        print("请手动运行：streamlit run app.py")
