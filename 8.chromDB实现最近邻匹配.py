from langchain.prompts.example_selector import SemanticSimilarityExampleSelector   
from langchain.prompts import FewShotPromptTemplate,PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import torch

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

examples = [
    {
        "text": "小米手机的创始人什么时候出生", 
     "answer": """
     这里是否需要跟进问题：是的。
     追问：小米的手机创始人是谁？
     回答：小米手机的创始人是雷军。
     追问：雷军什么时候出生？
     中间答案：雷军出生于1969年8月16日。
     所以最终答案是：1969年8月16日。
     """},
     {
         "text": "乾隆和曹操谁活的更久？",
         "answer": """
         这里是否需要跟进问题：是的。
         追问：乾隆和曹操的出生日期是什么？
         回答：乾隆出生于1711年9月25日，曹操出生于155年7月18日。
         追问：乾隆和曹操什么时候去世的？
         中间答案：乾隆去世于1799年2月7日，曹操去世于220年3月15日。
         回答：乾隆活了88岁，曹操活了66岁。
         所以最终答案是：乾隆活的更久。
         """
         
     }
]


example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,          # 第一个参数是示例
    embeddings=embeddings,      # 第二个参数是嵌入模型
    vectorstore_cls=Chroma,     # 第三个参数是向量存储类
    k=1                         # 其他参数
)

query = "李白和白居易谁活的更久？"
# select_examples = example_selector.select_examples({"text": query})
# print(
#     f'与输入最相似的示例：{select_examples}'
# )

# 示例提示词模板
examples_prompt = PromptTemplate(
    input_variables=["text", "answer"],
    template= "text: {text} \n answer: {answer} \n"
)

#动态提示词
prompt = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt=examples_prompt,
    prefix="根据案例的方式回答问题。 \n",
    suffix="text: {text} ",
    input_variables=["text"],
)

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"text": query}))   



