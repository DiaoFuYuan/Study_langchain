from langchain.retrievers import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader,PyPDFDirectoryLoader
from langchain_openai import ChatOpenAI
from langchain.storage import InMemoryStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
import uuid
import os
import nltk

nltk.data.path.append(r"D:\ai\study\nltk_data")
embeddings = HuggingFaceEmbeddings(
	# 修改为你的BGE-M3模型路径
	model_name="D:/ai/study/bge-m3",
	model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
	# model_kwargs={"device": "cpu"},
	encode_kwargs={"normalize_embeddings": True}
)

llm = ChatOpenAI(
	model_name="DeepSeek-R1-32b", # 如：deepseek-chat
	openai_api_key="sk-4d1c2f1ab80049f1a55b1e2e45694cc5",
	openai_api_base="http://36.138.75.130:18001/v1",  # 如: https://api.deepseek.com
)

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
print(all_data)

faiss_vectorstore = FAISS.from_documents(all_data,
									 embedding=embeddings,
									 )
faiss_vectorstore.save_local(r"D:\ai\study\file\faiss_vectorstore")

faiss_store = InMemoryStore()  #内存存储向量
faiss_retriever = MultiVectorRetriever(vectorstore=faiss_vectorstore, #  向量数据库
									   byte_store=faiss_store, # 存储向量的数据库
									   search_kwargs={"k": 3},
									   id_key="doc_id") #  向量数据库中保存的向量id的键名

doc_ids = [str(uuid.uuid4()) for _ in all_data]
print(doc_ids)

all_data_text_splitter = CharacterTextSplitter(
	separator="\n\n",
	chunk_size=1000,
	chunk_overlap=100,
	length_function=len,
	is_separator_regex=False,
)

# 为每个分割模块创建分割后的文档唯一id
sub_docs = []
for i ,doc in enumerate(all_data):
	_id = doc_ids[i]
	_sub_docs = all_data_text_splitter.split_documents([doc])
	for _doc in _sub_docs:
		_doc.metadata["doc_id"] = _id
	sub_docs.extend(_sub_docs)
print(sub_docs)

# # 使用llm对文档内容进行摘要总结
# summary_chain = (
# 	{"doc": lambda x: x.page_content}
# 	| ChatPromptTemplate.from_template("""
# 		请总结如下文档内容：
# 		{doc}
# 	""")
# 	| llm
# 	| StrOutputParser()
# )
#
# summarys = summary_chain.batch(all_data,{"max_concurrency": 1})
# print(summarys)

# 结合自查询的检索方式
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

metadata_attr = [
	AttributeInfo(
		attr="source",
		description="文件来源",
		required=True
	)
]

faiss_vectorstore = FAISS.load_local(
    folder_path=r"D:\ai\study\file\faiss_vectorstore",
    embeddings=embeddings
)
