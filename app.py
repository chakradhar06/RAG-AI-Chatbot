import os
import tempfile
import streamlit as st

from huggingface_hub import InferenceClient
from langchain.llms.base import LLM
from typing import Optional, List
from pydantic import PrivateAttr

from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------------- Hugging Face LLM Wrapper ----------------------

class HuggingFaceInferenceLLM(LLM):
    model_id: str
    temperature: float = 0.1
    max_new_tokens: int = 512
    _client: InferenceClient = PrivateAttr()

    def __init__(self, model_id: str, token: str, temperature: float = 0.1, max_new_tokens: int = 512):
        super().__init__(model_id=model_id, temperature=temperature, max_new_tokens=max_new_tokens)
        self._client = InferenceClient(model=model_id, token=token)

    @property
    def _llm_type(self) -> str:
        return "huggingface-inference-client"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._client.text_generation(
            prompt=prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            stop_sequences=stop or [],
        )
        return response.strip()
    

# ---------------------- Set API Token and Model ----------------------

HUGGINGFACE_TOKEN = "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"
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

llm = HuggingFaceInferenceLLM(
    model_id=MODEL_ID,
    token=HUGGINGFACE_TOKEN,
    temperature=0.7,
    max_new_tokens=512
)

# ---------------------- Streamlit UI Configuration ----------------------

st.set_page_config(page_title="Chatbot with RAG", layout="wide")
st.sidebar.title("Chatbot Settings")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# ---------------------- Load PDF and Build Index ----------------------

@st.cache_resource
def load_pdf(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_pdf_path = temp_file.name

        loaders = [PyPDFLoader(temp_pdf_path)]
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2'),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        ).from_loaders(loaders)

        os.remove(temp_pdf_path)
        return index
    except Exception as e:
        st.error(f"Failed to load document: {e}")
        return None

index = None
if uploaded_file:
    with st.spinner("Processing document..."):
        index = load_pdf(uploaded_file)
        if index:
            st.sidebar.success("Document Loaded Successfully!")

# ---------------------- Chat UI and Logic ----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("AI Chatbot with RAG")

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input("Ask me anything about the document...")

if prompt and index:
    with st.spinner("Thinking..."):
        try:
            retriever = index.vectorstore.as_retriever()
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=retriever,
                input_key='question'
            )
            response = chain.run(prompt)

            st.chat_message("user").markdown(prompt)
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------- Clear Chat Option ----------------------

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.sidebar.success("Chat history cleared!")

# ---------------------- End ----------------------

# Required pip installs:
# pip install streamlit langchain watchdog langchain_community sentence-transformers pypdf

# Models:
# meta-llama/Llama-3.2-1B-Instruct
# mistralai/Mistral-7B-Instruct-v0.2
# Qwen/Qwen2.5-Coder-32B-Instruct
