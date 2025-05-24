from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# 初始化LLM
openai_LLM = ChatOpenAI(
    model_name="deepseek-chat", # 如：deepseek-chat
    openai_api_key="sk-4d1c2f1ab80049f1a55b1e2e45694cc5",
    openai_api_base="https://api.deepseek.com",  # 如: https://api.deepseek.com 
    streaming=True
)

# 系统提示
system_prompt = "你是一个严谨的ai助手，请根据用户的问题,给出严谨简约的回答"

# 创建消息历史记录列表
chat_history = []

# 函数：获取带有历史记录的提示模板
def get_prompt_with_history(user_input):
    messages = [
        ("system", system_prompt)
    ]
    # 添加历史消息
    for message in chat_history:
        if isinstance(message, HumanMessage):
            messages.append(("human", message.content))
        elif isinstance(message, AIMessage):
            messages.append(("ai", message.content))
    
    # 添加当前用户输入
    messages.append(("human", user_input))
    response = ChatPromptTemplate.from_messages(messages)
    
    return response

# 聊天函数
def chat(user_input):
    # 创建包含历史记录的提示模板
    prompt = get_prompt_with_history(user_input)
    
    # 创建链并执行
    chain = prompt | openai_LLM
    
    # 添加用户消息到历史记录
    chat_history.append(HumanMessage(content=user_input))
    
    # 流式响应
    response_content = ""
    for chunk in chain.stream({}):
        chunk_text = chunk.content
        print(chunk_text, end="", flush=True)
        response_content += chunk_text
    
    # 将AI回复添加到历史记录
    chat_history.append(AIMessage(content=response_content))
    
    return response_content

# 演示历史对话
print("第一次对话")
print(chat_history)
response1 = chat("你是谁")
print(response1)

print("\n第二次对话")
print(chat_history)
response2 = chat("你能记住我之前问了什么吗？")
print(response2)

print("\n第三次对话")
print(chat_history)
response3 = chat("我们一共聊了几轮？")
print(response3)





