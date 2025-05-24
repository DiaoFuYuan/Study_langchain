from langchain_community.tools.tavily_search import TavilySearchResults
import os
import torch
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

os.environ["TAVILY_API_KEY"] = "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"


llm = ChatOpenAI(
	model_name="deepseek-chat", # 如：deepseek-chat
	openai_api_key="sk-4d1c2f1ab80049f1a55b1e2e45694cc5",
	openai_api_base="https://api.deepseek.com",  # 如: https://api.deepseek.com
)

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是一个有用的AI助手。使用提供的工具来回答用户问题。"),
#     ("user", "{input}"),
#     ("assistant", "{agent_scratchpad}")  
# ])
prompt = hub.pull("hwchase17/openai-functions-agent")

search = TavilySearchResults(max_results=1)
tools = [search]

from langchain.agents import create_tool_calling_agent
agent = create_tool_calling_agent(llm,tools,prompt)
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)
print(agent_executor.invoke({"input":"遂宁今天的天气如何"}))





# embeddings = HuggingFaceEmbeddings(
# 	model_name="D:/ai/study/bge-m3",
# 	model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
# 	encode_kwargs={"normalize_embeddings": True}
# )

# faiss_vectorstore = FAISS.from_documents(search_result,embeddings)
# faiss_vectorstore.save_local(r"D:\ai\study\ceshi")










 
