from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

llm = ChatOpenAI(
    model_name="deepseek-chat",
    openai_api_key="sk-4d1c2f1ab80049f1a55b1e2e45694cc5",
    openai_api_base="https://api.deepseek.com",
)


llm_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有用的AI助手。你根据以下内容回答用户问题:"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
# 创建基本链，不包含记忆管理
chain = llm_prompt | llm | StrOutputParser()

# 包装链以自动管理聊天历史
message_history = ChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: message_history,
    input_messages_key="input",     # 输入消息的键名
    history_messages_key="chat_history"  # 历史消息的键名
)

# 使用唯一的会话ID
session_id = "user_123"

# 第一次对话
result = chain_with_history.invoke(
    {"input": "你好，我叫刁福与"},
    config={"configurable": {"session_id": session_id}}
)
print(result)

# 第二次对话 - 会自动使用之前的历史
result1 = chain_with_history.invoke(
    {"input": "我叫什么名字"},
    config={"configurable": {"session_id": session_id}}
)
print(result1)

# 第三次对话 - 继续自动使用之前的所有历史
result2 = chain_with_history.invoke(
    {"input": "你能记住我之前说过的话吗？"},
    config={"configurable": {"session_id": session_id}}
)
print(result2)
