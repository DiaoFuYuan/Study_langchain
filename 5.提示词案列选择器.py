from langchain.prompts import FewShotPromptTemplate,PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector,LengthBasedExampleSelector
from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector,MaxMarginalRelevanceExampleSelector
from langchain_community.vectorstores import FAISS



openai_LLM = ChatOpenAI(
    model_name="deepseek-chat", # 如：deepseek-chat
    openai_api_key="sk-4d1c2f1ab80049f1a55b1e2e45694cc5",
    openai_api_base="https://api.deepseek.com",  # 如: https://api.deepseek.com 
)

# 示例模板
examples = [
    {"text": "开心", "answer": "伤心"},
    {"text": "高", "answer": "矮"},
    {"text": "胖", "answer": "瘦"},
]

# 示例提示词
examples_prompt = PromptTemplate(
    input_variables=["text", "answer"],
    template= "text: {text} \n answer: {answer} \n"
)

# 示例选择器
example_selector = LengthBasedExampleSelector(
    # 可供选择的示例
    examples=examples,
    # 示例的提示词
    example_prompt=examples_prompt,
    # 示例的最大长度，如果超过这个长度，则不选择
    max_length=100
)

# 动态提示词
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=examples_prompt,
    prefix="给出每个输入的反义词",
    suffix="text: {text} \n answer: ",
    input_variables=["text"]
)

# 输出解析器
chain_llm = dynamic_prompt | openai_LLM | StrOutputParser()

# 执行
print(chain_llm.invoke({"text":"开心"}))



















