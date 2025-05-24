from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
import torch

# 定义一个使用本地嵌入模型的LangChain嵌入类
class CustomBGEEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, model_path="D:/ai/study/bge-m3", **kwargs):
        super().__init__(
            model_name=model_path,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            **kwargs
        )
    
    def embed_documents(self, texts):
        """嵌入文档列表"""
        return super().embed_documents(texts)
    
    def embed_query(self, text):
        """嵌入单个查询"""
        return super().embed_query(text)

# 创建一些示例数据
examples = [
    {"input": "如何做红烧肉？", "output": "准备五花肉，加入酱油、糖和其他调料，慢火炖煮至酥烂。"},
    {"input": "如何学习Python？", "output": "从基础语法开始，然后学习数据结构，接着实践小项目。"},
    {"input": "推荐一本科幻小说", "output": "《三体》是一部优秀的中国科幻小说，讲述了地球文明与三体文明的相遇。"},
    {"input": "如何练习英语口语？", "output": "可以找语伴交流，看英语影视作品，或参加口语培训班。"},
    {"input": "如何保持健康的生活方式？", "output": "均衡饮食，规律作息，适量运动，保持心情愉快。"},
    {"input": "什么是人工智能？", "output": "人工智能是研究如何使计算机模拟人类智能的一门科学。"},
    {"input": "如何做披萨？", "output": "准备面团，添加番茄酱和奶酪，再加上喜欢的配料，烤箱烘焙即可。"}
]

# 将示例转换为Document对象
example_docs = [
    Document(
        page_content=f"问题: {example['input']}\n回答: {example['output']}",
        metadata={"input": example["input"], "output": example["output"]}
    )
    for example in examples
]

def main():
    # 初始化本地嵌入模型
    embeddings = CustomBGEEmbeddings()
    
    print("嵌入模型加载完成!")
    
    # 创建最大边际相关性示例选择器
    example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
        examples=example_docs,  # 示例文档
        embeddings=embeddings,  # 嵌入模型
        vectorstore_cls=FAISS,  # 向量存储类
        k=3  # 返回前3个相关示例
    )
    
    # 测试查询
    query = "我想了解如何做中国菜"
    selected_examples = example_selector.select_examples({"input": query})
    
    print(f"\n查询: {query}")
    print("\n选择的最相关示例:")
    
    for i, example in enumerate(selected_examples, 1):
        print(f"\n示例 {i}:")
        print(f"问题: {example.metadata['input']}")
        print(f"回答: {example.metadata['output']}")
        
    # 另一个测试查询
    query2 = "我想学习编程"
    selected_examples2 = example_selector.select_examples({"input": query2})
    
    print(f"\n\n查询: {query2}")
    print("\n选择的最相关示例:")
    
    for i, example in enumerate(selected_examples2, 1):
        print(f"\n示例 {i}:")
        print(f"问题: {example.metadata['input']}")
        print(f"回答: {example.metadata['output']}")

if __name__ == "__main__":
    main() 