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

# 测试示例
if __name__ == "__main__":
    texts = ["这是一个测试句子", "这是另一个不同的句子", "这是两个句子"]
    embeddings = get_embeddings(texts)
    
    print(f"嵌入向量形状: {embeddings.shape}")
    
    # 计算两个句子的相似度
    similarity = embeddings[0] @ embeddings[1].T
    print(f"两个句子的相似度: {similarity}")
