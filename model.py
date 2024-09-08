from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
import os
from transformers import pipeline
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import time
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
import streamlit as st

load_dotenv()


try:
    # Try to get the API key from Streamlit secrets
     together_api_key = st.secrets["together_ai"]["api_key"]
     pinecone_api_key = st.secrets["pinecone_ai"]["api_key"]

except:
    # If running locally, fall back to environment variable
    together_api_key = os.getenv("TOGETHER_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not together_api_key:
    st.error("API key not found. Please set it up in Streamlit secrets or .env file.")
    st.stop()


st.set_page_config(page_title="LawGPT")
col1, col2, col3 = st.columns([1,4,1])
with col2:
    st.image("./assets/Black Bold Initial AI Business Logo.jpg")
st.markdown(
    """
     <style>
    .stApp, .ea3mdgi6{
      background-color:#000000;
    }
  div.stButton > button:first-child {
    background-color: #ffd0d0;
}
div.stButton > button:active {
    # background-color: #ff6262;
}
   div[data-testid="stStatusWidget"] div button {
        display: none;
        }
    
    .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    button[title="View fullscreen"]{
    visibility: hidden;}
    button:first-child{
    background-color : transparent !important;
    }
  </style>
""",
  unsafe_allow_html=True,
)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={"trust_remote_code":True, "revision":"289f532e14dbbbd5a04753fa58739e9ba766f3c7"}
    )

@st.cache_resource
def load_vector_store():
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "legalaivectors"
    index = pc.Index(index_name)
    embeddings = load_embeddings()
    return PineconeVectorStore(index=index, embedding=embeddings)

@st.cache_resource
def load_llm():
    return Together(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.5,
        max_tokens=1024,
        together_api_key=together_api_key
    )

def reset_conversation():
  st.session_state.messages = []
  st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferWindowMemory(k=2, memory_key="chat_history",return_messages=True) 

vector_store = load_vector_store()
db_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

prompt_template = """<s>[INST] You are a legal chatbot specializing in Indian Penal Code queries. You must only provide answers related to the Indian Penal Code. If the question is unrelated to the Indian Penal Code, respond with: "I can only answer questions related to the Indian Penal Code."
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question', 'chat_history'])

llm = load_llm()

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db_retriever,
    memory=ConversationBufferWindowMemory(k=2, memory_key="chat_history",return_messages=True),
    combine_docs_chain_kwargs={'prompt': prompt}
)
for message in st.session_state.get("messages", []):
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

input_prompt = st.chat_input("Say something")


if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role":"user","content":input_prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...",expanded=True):
            result = qa.invoke(input=input_prompt)

            message_placeholder = st.empty()

            full_response = "⚠️ **_Note: Information provided may be inaccurate._** \n\n\n"
        for chunk in result["answer"]:
            full_response+=chunk
            time.sleep(0.02)
            
            message_placeholder.markdown(full_response+" ▌")
        st.button('Reset All Chat 🗑️', on_click=reset_conversation)

    st.session_state.messages.append({"role":"assistant","content":result["answer"]})



    