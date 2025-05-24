from langchain.prompts import FewShotPromptTemplate,PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector,LengthBasedExampleSelector
from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector,MaxMarginalRelevanceExampleSelector
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel
import torch
import os

examples = [
    {"text": "小米手机的创始人什么时候出生", 
     "answer": """
     这里是否需要跟进问题：是的。
     追问：小米的手机创始人是谁？
     回答：小米手机的创始人是雷军。
     追问：雷军什么时候出生？
     中间答案：雷军出生于1969年8月16日。
     所以最终答案是：1969年8月16日。
     """},
    
]


# 示例提示词
examples_prompt = PromptTemplate(
    input_variables=["text", "answer"],
    template= "text: {text} \n answer: {answer} \n"
)

# 示例提示词模板
prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=examples_prompt,
    suffix="text: {text} \n answer:",
    input_variables=["text"],
)


llm = ChatOpenAI(
    model_name="deepseek-chat", # 如：deepseek-chat
    openai_api_key="sk-4d1c2f1ab80049f1a55b1e2e45694cc5",
    openai_api_base="https://api.deepseek.com",  # 如: https://api.deepseek.com
)


output_parser = StrOutputParser()
chain = prompt_template | llm | output_parser

print(chain.invoke({"text":"诗仙的出生日期是多久？"})) 