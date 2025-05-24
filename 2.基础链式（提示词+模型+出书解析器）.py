from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llmprompt = ChatPromptTemplate.from_template("你是一个幽默的AI助手")

llm = ChatOpenAI(
    model_name="deepseek-chat",
    openai_api_key="sk-4d1c2f1ab80049f1a55b1e2e45694cc5",
    openai_api_base="https://api.deepseek.com",
)

parser = StrOutputParser()

chain = llmprompt | llm | parser

for chunk in chain.stream({"text": "你好"}):
    print(chunk, end="", flush=True)







