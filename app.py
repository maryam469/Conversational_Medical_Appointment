import streamlit as st
import os
import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain.chains.conversation.memory import ConversationBufferMemory

from langchain.chains import ConversationChain
from langchain.chains import ConversationChain


# streamlit App setup

st.set_page_config(page_title="Medical Chatbot", layout="wide")
# Title
st.markdown("<h1 style='text-align:center;'>Medical Appointment Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Talk to your AI assistant for booking and advice.</p><hr>", unsafe_allow_html=True)


##create chat folder if not exist
if not os.path.exists("chats"):
    os.makedirs("chats")

    ##sidebar controls

    st.sidebar.header("⚙️ Settings")


# API Key Input
user_api_key = st.sidebar.text_input("🔑 Enter your GROQ API Key", type="password")

model_name = st.sidebar.selectbox(
    "Select Groq Model",
    ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", "llama-3.1-8b-instant"]
)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)


#sidebar: Load previous chats
chat_files = sorted([f for f in os.listdir("chats") if f.endswith(".json")])
selected_chat = st.sidebar.selectbox("📂 Load Previous Chat", [""] + chat_files)


if st.sidebar.button("📤 Load Selected Chat") and selected_chat:
    with open(f"chats/{selected_chat}", "r") as f:
        st.session_state.history = json.load(f)
    st.success(f"Loaded: {selected_chat}")


# Sidebar: New chat button
if st.sidebar.button("🆕 New Chat"):
    st.session_state.history = []
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.success("✨ New chat started!")


##initialize memory $ history
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "history" not in st.session_state:
    st.session_state.history = []


##check Api key

if not user_api_key:
    st.warning("⚠️ Please enter your GROQ API Key in the sidebar to start chatting.")
else:
    ##user input

    user_input = st.chat_input("You : ")

    if user_input and isinstance(user_input, str):
        st.session_state.history.append(("user", user_input))


        ##LLM initialization
        llm = ChatGroq(
            model_name=model_name,
            max_tokens=max_tokens,
            groq_api_key = user_api_key
        )

        ##Build conversation chain with memory
        conv = ConversationChain(
            llm = llm,
            memory = st.session_state.memory,
            verbose = False
        )
        try:
            if user_input.lower().strip() in ["hi", "hello", "hey"]:
                ai_response = "Hello! 👋 It's nice to meet you. Would you like to book an appointment? 😊"
            else:
                ai_response = conv.predict(input=user_input)
        except Exception as e:
            ai_response = f"⚠️ Error: {str(e)}"




        ##ADD Ai response to history
        st.session_state.history.append(("assistant", ai_response))


        # 12. Auto-save chat in unique file
        chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
        with open(f"chats/chat_{chat_id}.json", "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, ensure_ascii=False)

for role, text in st.session_state.history:
    if role == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(
                f"<div style='background-color:#E6E6FA; color:black; padding:10px; border-radius:10px;'>{text}</div>",
                unsafe_allow_html=True
            )
    elif role == "assistant":
        with st.chat_message("assistant", avatar="💬"):
            st.markdown(
                f"<div style='background-color:#444654; color:white; padding:10px; border-radius:10px;'>{text}</div>",
                unsafe_allow_html=True
            )
