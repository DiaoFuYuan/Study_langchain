from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = ChatOpenAI(
    model_name="deepseek-chat", # 如：deepseek-chat
    openai_api_key="sk-4d1c2f1ab80049f1a55b1e2e45694cc5",
    openai_api_base="https://api.deepseek.com",  # 如: https://api.deepseek.com
)


output_parser = StrOutputParser()
prompt_template = ChatPromptTemplate.from_template("{text}")
chain = prompt_template | llm | output_parser

print(chain.invoke("你好")) 
