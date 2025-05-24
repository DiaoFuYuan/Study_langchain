from langchain.prompts import FewShotPromptTemplate,PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector,LengthBasedExampleSelector
from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector,MaxMarginalRelevanceExampleSelector
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel
import torch
import os


def get_embeddings(texts):
    """获取文本嵌入向量"""
    # 编码文本

    model_path = "D:/ai/study/bge-m3"  # 注意这里使用正斜杠

    # 加载分词器和模型，明确指定从本地加载
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)

    # 将模型移动到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    encoded_input = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    ).to(device)
    
    # 计算嵌入向量
    with torch.no_grad():
        model_output = model(**encoded_input)
        # 使用最后一层隐藏状态的平均值作为文本表示
        embeddings = model_output.last_hidden_state[:, 0]
        # 归一化嵌入向量
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()


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

# 定义最大余炫案例选择器
example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    # 可供选择的示例
    examples=examples,
    # 嵌入模型
    embedding=get_embeddings,
    # 示例的最大长度，如果超过这个长度，则不选择
    k=2,
    # 向量存储
    vectorstore=FAISS
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



















