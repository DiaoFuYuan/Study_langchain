from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader,PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import nltk
import os

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

# UnstructuredFileLoader 可以加载多种文件类型，包括txt,pdf,docx,csv,excel,pptx,html,xml,json,yaml,yml,jsonl,jsonl.gz,jsonl.bz2,jsonl.zip,jsonl.tar,jsonl.tar.gz,jsonl.tar.bz2,jsonl.tar.zip,jsonl.tar.tar,jsonl.tar.tar.gz,jsonl.tar.tar.bz2,jsonl.tar.tar.zip
# excel_loader = UnstructuredLoader(file_path=r"D:\ai\study\file\1.xlsx",encoding="utf-8")
# data_excel = excel_loader.load()
# print(data_excel)

# word_loader = UnstructuredLoader(file_path=r"D:\ai\study\file\处罚决定书（四川华筑联投置业有限公司）.docx",encoding="utf-8")
# data_word = word_loader.load()
# print(data_word)

# 加载单个pdf文件
# pdf_loader = PyPDFLoader(file_path=r"D:\ai\study\file\处罚决定书（四川华筑联投置业有限公司）.pdf")
# data_pdf = pdf_loader.load_and_split()
# 加载文件夹下多个pdf文件
# pdf_loader = PyPDFDirectoryLoader(path=r"D:\ai\study\file")
# data_pdf = pdf_loader.load()

data_pdf = []
data_doc = []
path = r"D:\ai\study\file"
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    if file.endswith('.pdf'):
        print(f"正在处理PDF文件: {file}")
        pdf_loader = PyPDFLoader(file_path=file_path)
        data_pdf.extend(pdf_loader.load_and_split())
        print(f"处理完成: {file}")
    else:
        print(f"正在处理其他类型文件: {file}")
        doc_loader = UnstructuredLoader(file_path=file_path, encoding="utf-8",show_progress=True)
        data_doc.extend(doc_loader.load())
        print(f"处理完成: {file}")

all_data = data_pdf + data_doc
# 处理文件夹下的所有文件
faiss_vectorstore = FAISS.from_documents(all_data,
                                     embedding=embeddings)

# 检索匹配
# 余弦相似性检索（Similarity Search）是一种用于文本检索的方法，它的目标是在给定一个查询时，找到与查询最相似的文档。
# 最大边际相关性检索（Maximum Marginal Relevance，MMR）是一种用于文本检索的方法，它的目标是在给定一个查询时，找到与查询相关但与其他检索结果之间的相关性最小的文档。
retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2},search_type="mmr")
# retriever = faiss_vectorstore.as_retriever(search_kwargs={"score_threshold": 0.3},search_type="similarity_score_threshold")
docs = retriever.get_relevant_documents(r"四川省遂宁市河东新区九宗书院路366号附14、15号")
print(docs)

setup_and_retrieval = RunnableParallel({
    "context":retriever,
    "question":RunnablePassthrough()
})

prompt = ChatPromptTemplate.from_template("""
你是一个专业的数据分析师，现在需要你根据用户的问题，从给定的数据中检索出相关的信息。
问题：{question}
数据：{context}
请根据问题和数据，给出详细的分析和回答。
""")

chain = setup_and_retrieval | prompt | llm | StrOutputParser()
print(chain.invoke("四川省遂宁市河东新区九宗书院路366号附14、15号"))
