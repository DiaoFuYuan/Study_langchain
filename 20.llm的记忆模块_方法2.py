from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# 初始化LLM
llm = ChatOpenAI(
    model_name="deepseek-chat",
    openai_api_key="sk-4d1c2f1ab80049f1a55b1e2e45694cc5",
    openai_api_base="https://api.deepseek.com",
)

# 创建记忆对象
memory = ConversationBufferMemory()

# 创建对话链，自动管理历史
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # 设置为True可以看到详细过程
)

# 第一次对话
response1 = conversation.predict(input="你好，我叫刁福与")
print(response1)

# 第二次对话 - 自动使用之前的历史
response2 = conversation.predict(input="我叫什么名字")
print(response2)

# 第三次对话 - 继续使用所有历史
response3 = conversation.predict(input="你能记住我之前说过的话吗？")
print(response3) 