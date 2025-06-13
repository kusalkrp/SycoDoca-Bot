import os
from dotenv import load_dotenv

import streamlit as st
from streamlit_chat import message
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

from openai import OpenAI

# Load env
load_dotenv()

# OpenAI-compatible Hugging Face client
client = OpenAI(
    base_url="https://router.huggingface.co/hyperbolic/v1",
    api_key=os.environ["HF_TOKEN"],
)

# Streamlit config
st.set_page_config(page_title="Mental Health Chatbot", page_icon=":robot_face:")

# Embeddings
def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

embeddings = download_hugging_face_embeddings()
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index_name = "mental-chatbot"
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

# Custom QA class
class CustomRetrievalQA:
    def __init__(self, retriever):
        self.retriever = retriever

    def generate_response(self, user_input):
        # Use new .invoke() instead of deprecated method
        retrieved_docs = self.retriever.invoke(user_input)
        context = " ".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""Based on the context and user's answers, generate a comprehensive mental health response. 
It should be 100–150 words, formatted into 2–3 paragraphs.

Context:
{context}

User's Answers:
{user_input}

Response:"""

        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        return completion.choices[0].message.content

retriever = docsearch.as_retriever(search_kwargs={"k": 2})
custom_qa = CustomRetrievalQA(retriever=retriever)

# Predefined questions
predefined_questions = [
    "Can you describe any recent changes in your mood or emotions?",
    "How has your energy level fluctuated over the past few weeks?",
    "How often do you feel anxious or worried, and what typically triggers these feelings?",
    "How would you describe your sleep patterns lately?",
    "Have you noticed any changes in your appetite or eating habits?",
    "How do you generally feel in social situations?",
    "Can you describe any impulsive behaviors you've noticed?",
    "Have you experienced moments of feeling disconnected from reality?",
    "Are you dealing with any physical symptoms with no clear medical explanation?",
    "Have you gone through any traumatic events that still affect you?",
]

# Session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "user_answers" not in st.session_state:
    st.session_state["user_answers"] = []
if "question_index" not in st.session_state:
    st.session_state["question_index"] = 0
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

# Initial question
if st.session_state["question_index"] == 0 and not st.session_state["messages"]:
    st.session_state["messages"].append({"sender": "bot", "content": predefined_questions[0]})

# Handle bot logic
def chat_with_bot(user_answers):
    user_input = " ".join(user_answers)
    return custom_qa.generate_response(user_input)

def submit_user_input():
    user_input = st.session_state["user_input"]
    if user_input:
        st.session_state["messages"].append({"sender": "user", "content": user_input})
        st.session_state["user_answers"].append(user_input)
        st.session_state["user_input"] = ""
        st.session_state["question_index"] += 1
        if st.session_state["question_index"] < len(predefined_questions):
            next_q = predefined_questions[st.session_state["question_index"]]
            st.session_state["messages"].append({"sender": "bot", "content": next_q})
        else:
            final_response = chat_with_bot(st.session_state["user_answers"])
            st.session_state["messages"].append({"sender": "bot", "content": final_response})

# Page title
st.title("SycoDoca")
st.title("Mental Health Chatbot")

# Chat UI
for i, msg in enumerate(st.session_state["messages"]):
    key = f"{msg['sender']}_{i}"
    message(msg["content"], is_user=(msg["sender"] == "user"), key=key)

if st.session_state["question_index"] < len(predefined_questions):
    st.text_input("Your answer:", key="user_input", on_change=submit_user_input)
else:
    if st.button("Start a new conversation"):
        st.session_state["messages"] = []
        st.session_state["user_answers"] = []
        st.session_state["question_index"] = 0
        st.session_state["user_input"] = ""
        st.session_state["messages"].append({"sender": "bot", "content": predefined_questions[0]})

# Footer
st.markdown(
    """
    <div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%);
    padding: 10px; font-size: 12px; color: #666;">
    Designed and Developed by Pahanmi (PVT) Ltd
    </div>
    """,
    unsafe_allow_html=True,
)

# Styling
st.markdown(
    """
    <style>
    body { background-color: white; }
    .stTextInput > div > input {
        padding: 12px;
        font-size: 16px;
        border-radius: 10px;
        border: 2px solid #ccc;
        background-color: #f7f7f7;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    .stButton>button {
        padding: 12px 24px;
        font-size: 16px;
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        border: none;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stButton>button:before {
        content: "\\1F4AC";
        margin-right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
