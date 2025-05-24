import streamlit as st
import tempfile
import os
from langchain.memory import ConversationBufferMemory  # 内存管理
from langchain_community.chat_message_histories import StreamlitChatMessageHistory  # 消息历史
from langchain_community.document_loaders import TextLoader  # 文本加载
from langchain_huggingface import HuggingFaceEmbeddings  # 嵌入模型
from langchain_chroma import Chroma  # 向量数据库
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate  # 提示词
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本分割
from langchain.agents import create_react_agent, AgentExecutor  # 智能体
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler  # 流式输出
from langchain_openai import ChatOpenAI  # 语言模型
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader,PyPDFDirectoryLoader
import nltk
import torch
from langchain_core.documents import Document
import pandas as pd
nltk.data.path.append(r"D:\ai\study\nltk_data")

embeddings = HuggingFaceEmbeddings(
    # 修改为你的BGE-M3模型路径
    model_name="D:/ai/study/bge-m3",
    # model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

llm = ChatOpenAI(
    model_name="deepseek-chat", # 如：deepseek-chat
    openai_api_key="sk-4d1c2f1ab80049f1a55b1e2e45694cc5",
    openai_api_base="https://api.deepseek.com",  # 如: https://api.deepseek.com
)


# 设置Streamlit应用的页面标题和布局
st.set_page_config(page_title="多文档智能问答系统",layout="wide")
# 设置应用的标题
st.title("多文档智能问答系统")
st.markdown("### 基于RAG技术的多格式文档知识库问答")
st.markdown("上传您的文档（支持PDF、Excel、Word、TXT等格式），系统将自动构建知识库并回答您的问题。")

# 上传多种类型文件，允许上传多个文件
uploaded_files = st.sidebar.file_uploader(
    label="上传文档（支持PDF、Excel、TXT等格式）",
    type=["txt", "pdf", "docx", "xlsx", "xls", "csv"],
    accept_multiple_files=True
)
# 如果没有上传文件，提示用户上传文件并停止运行
if not uploaded_files:
    st.info("请先上传文档（支持PDF、Excel、TXT等格式）。")
    st.stop()


def process_excel_file(file_path):
    """处理Excel或CSV文件，读取每行数据并转换为Document对象"""
    documents = []
    try:
        # 根据文件类型选择不同的读取方法
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
        else:  # Excel文件
            df = pd.read_excel(file_path)
            
        file_name = os.path.basename(file_path)
        # 处理每一行数据
        for index, row in df.iterrows():
            # 将行数据转换为字符串
            row_content = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            # 创建Document对象
            doc = Document(
                page_content=row_content,
                metadata={
                    "source": file_path,
                    "row_index": index,
                    "file_name": file_name
                }
            )
            documents.append(doc)
        return documents
    except Exception as e:
        st.sidebar.error(f"处理文件时出错: {e}")
        return []

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # 处理上传的文档
    data_pdf = []
    data_doc = []
    data_excel = []
    
    # 创建临时目录存储上传的文件
    with tempfile.TemporaryDirectory() as temp_dir:
        # 保存上传的文件到临时目录
        temp_files = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_files.append(file_path)
            
            # 根据文件类型进行处理
            if file_path.lower().endswith('.pdf'):
                st.sidebar.write(f"正在处理PDF文件: {uploaded_file.name}")
                pdf_loader = PyPDFLoader(file_path=file_path)
                data_pdf.extend(pdf_loader.load_and_split())
                st.sidebar.write(f"处理完成: {uploaded_file.name}")
            elif file_path.lower().endswith(('.xlsx', '.xls', '.csv')):
                st.sidebar.write(f"正在处理Excel/CSV文件: {uploaded_file.name}")
                data_excel.extend(process_excel_file(file_path))
                st.sidebar.write(f"处理完成: {uploaded_file.name}")
            else:
                st.sidebar.write(f"正在处理文本文件: {uploaded_file.name}")
                try:
                    text_loader = TextLoader(file_path=file_path, encoding="utf-8")
                    data_doc.extend(text_loader.load())
                except Exception as e:
                    st.sidebar.error(f"处理文件 {uploaded_file.name} 时出错: {e}")
                    # 尝试使用UnstructuredLoader作为备选
                    try:
                        doc_loader = UnstructuredLoader(file_path=file_path, encoding="utf-8")
                        data_doc.extend(doc_loader.load())
                    except Exception as e2:
                        st.sidebar.error(f"备选处理方法也失败: {e2}")
                st.sidebar.write(f"处理完成: {uploaded_file.name}")
                
        docs = data_pdf + data_doc + data_excel
        
        if not docs:
            st.error("没有成功处理任何文档，请检查上传的文件格式是否正确。")
            st.stop()

        # 进行文档分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
 
        # 使用HuggingFace的向量模型生成文档的向量表示
        embeddings = HuggingFaceEmbeddings(
            model_name="D:/ai/study/bge-m3",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )

        # 创建向量数据库
        vectorstore = Chroma.from_documents(split_docs, embeddings)
        # 返回向量数据库的检索器
        return vectorstore.as_retriever()

# 创建一个检索器
retriever = configure_retriever(uploaded_files)

# 如果session_state中没有消息记录或用户点击了清空聊天记录按钮，则初始化消息记录
if"messages" not in st.session_state or st.sidebar.button("清空聊天记录"):
    st.session_state["messages"]= [{"role":"assistant","content":"您好，我是文档问答助手"}]

#加载历史聊天记录
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# 创建检索工具
from langchain.tools.retriever import create_retriever_tool
tool = create_retriever_tool(retriever, name="文档检索工具", description="用于检索文档中的信息，并基于检索到的信息回答用户的问题")

# 创建智能体
tools = [tool]
# 创建聊天消息历史对话记录
msgs = StreamlitChatMessageHistory()

# 创建记忆
memory = ConversationBufferMemory(chat_memory=msgs, memory_key="chat_history", output_key="output",return_messages=True)

# 创建模板
instructions = """
您是一个设计用于査询文档来回答问题的代理。
您可以使用文档检索工具，并基于检索内容来回答问题您可能不查询文档就知道答案，但是您仍然应该查询文档来获得答案。
如果您从文档中找不到任何信息用于回答问题，则只需返回"抱歉，这个问题我还不知道。"作为答案。
"""
# 基础提示词模板
base_prompt_template = """
{instructions}
TOOLS:
------
You have access to the following tools:
{tools}
To use a tool,please use the following format:
ZWJ'''
Thought:Do Ineed to use a tool? Yes
Action:the action to take, should be one of [{tool_names}]
Action Input: {input}
Observation:the result of the action
ZWJ'''

When you have a response to say to the Human, or if you do not need to use a tool, you MusT use the format:

ZWJ'''

Thought:Do Ineed to use a tool? No
Final Answer:[your response here]

ZWJ'''

Begin!
Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}

"""

# 将基础提示词模板转换为PromptTemplate对象
base_prompt = PromptTemplate.from_template(base_prompt_template)

# 创建部分填充的提示词模板
prompt = base_prompt.partial(instructions=instructions)


agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    verbose=True
)

agent_executor = AgentExecutor(agent=agent, tools=tools,memory=memory,handle_parse_errors="没有从知识库中找到相关信息", verbose=True)

#创建聊天输出入框
user_input = st.chat_input(placeholder="请输入您的问题")

  #如果有用户输入的查询
if user_input:
    #添加用户消息到session_state
    st.session_state.messages.append({"role":"user","content": user_input})
    #显示用户消息
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        # 创建Streamlit回调处理器
        st_cb =StreamlitCallbackHandler(st.container())
        # agent执行过程日志回调显示在Streamlit container(如思考、选择工具、执行査询、观察结果等)
        config ={"callbacks":[st_cb]}
        #执行Agent并获取响应
        response = agent_executor.invoke({"input": user_input}, config=config)
        #添加助手消息到session_state
        st.session_state.messages.append({"role": "assistant", "content": response["output"]})









