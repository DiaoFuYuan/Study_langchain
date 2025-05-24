import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory

# 设置页面标题
st.title("智能聊天助手")

# 初始化聊天历史
history = StreamlitChatMessageHistory(key="chat_messages")

# 初始化LLM
llm = ChatOpenAI(
    model_name="deepseek-chat",
    openai_api_key="sk-4d1c2f1ab80049f1a55b1e2e45694cc5",
    openai_api_base="https://api.deepseek.com",
)

# 创建记忆对象，使用StreamlitChatMessageHistory
memory = ConversationBufferMemory(
    chat_memory=history,
    return_messages=True,
    memory_key="chat_history"
)

# 创建提示模板
prompt = ChatPromptTemplate.from_template("""
你是一个有用的AI助手。你根据以下内容回答用户问题:
历史对话
{chat_history}

当前对话
{input}                                                     
""")

# 创建链
chain = prompt | llm | StrOutputParser()

# 显示聊天历史
for msg in history.messages:
    if msg.type == "human":
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

# 获取用户输入
if user_input := st.chat_input("请输入您的问题..."):
    # 显示用户输入
    st.chat_message("user").write(user_input)
    
    # 获取历史记录
    chat_history = memory.load_memory_variables({}).get("chat_history", "")
    
    # 生成回复
    with st.chat_message("assistant"):
        response = chain.invoke({"input": user_input, "chat_history": chat_history})
        st.write(response)
    
    # 保存对话到历史
    history.add_user_message(user_input)
    history.add_ai_message(response) 