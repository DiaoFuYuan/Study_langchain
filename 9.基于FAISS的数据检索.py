from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import torch
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

embeddings = HuggingFaceEmbeddings(
    # 修改为你的BGE-M3模型路径
    model_name="D:/ai/study/bge-m3",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

llm = ChatOpenAI(
    model_name="deepseek-chat", # 如：deepseek-chat
    openai_api_key="sk-4d1c2f1ab80049f1a55b1e2e45694cc5",
    openai_api_base="https://api.deepseek.com",  # 如: https://api.deepseek.com
)


faiss_vectorstore = FAISS.from_texts(["刁福元是一个四川移动的员工","李华喜欢吃香蕉","西南名族大学是成都的一所学校。"],
                                     embedding=embeddings)

# 检索匹配
retriever = faiss_vectorstore.as_retriever()
# retriever_result = retriever.get_relevant_documents("刁福元是谁")


prompt = ChatPromptTemplate.from_template("""
你是一个专业的数据分析师，现在需要你根据用户的问题，从给定的数据中检索出相关的信息。
问题：{question}
数据：{context}
请根据问题和数据，给出详细的分析和回答。
""")

# 并行处理问题和数据
setup_and_retrieval = RunnableParallel({
    "context":retriever,
    "question":RunnablePassthrough()
})

chain = setup_and_retrieval | prompt | llm | StrOutputParser()

result = chain.invoke("刁福元是谁")

print(result)


