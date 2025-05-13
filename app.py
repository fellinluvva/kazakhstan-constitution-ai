from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA
import streamlit as st
import os

# Set USER_AGENT
os.environ["USER_AGENT"] = "KazakhstanAI/1.0 (https://github.com/fellinluvva)"

# Set up embeddings and vector store
embedding = OllamaEmbeddings(model="nomic-embed-text")
DB_DIR = "./chroma_db"
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding)
retriever = vector_db.as_retriever()
llm = OllamaLLM(model="llama3")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)



# Process uploaded files
def process_documents(files):
    for file in files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(file.name) if file.name.endswith(".pdf") else TextLoader(file.name)
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        vector_db.add_documents(chunks)

# Answer user's question
def ask_question(question):
    return qa_chain.run(question)

# Streamlit UI
st.set_page_config(page_title="Kazakhstan Constitution AI Assistant")
st.title("Constitution AI Assistant")


uploaded_files = st.file_uploader("Upload your PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
if uploaded_files:
    with st.spinner("Processing uploaded documents..."):
        process_documents(uploaded_files)
    st.success("Files processed and added to the knowledge base.")

user_input = st.chat_input("Ask a question about uploaded docs...")
if user_input:
    st.write("You asked:", user_input)
    with st.spinner("Thinking..."):
        response = ask_question(user_input)
        st.markdown(f"**AI Response:** {response}")
