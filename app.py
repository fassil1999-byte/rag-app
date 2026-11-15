import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load keys from .env if present
load_dotenv()

# Read credentials directly from environment variables
api_key = os.getenv("OPENAI_API_KEY", "")
enable_tracing = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true"
langsmith_key = os.getenv("LANGCHAIN_API_KEY", "")
langsmith_project = os.getenv("LANGCHAIN_PROJECT", "rag-streamlit-app")

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="wide")
st.title("📄 Chat with Your Documents (OpenAI + LangSmith)")

# --- SIDEBAR: FILE UPLOAD ONLY ---
with st.sidebar:
    st.header("Upload Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    process_button = st.button("Process & Index Files")

# Helper function to sync environment variables
def configure_env():
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if enable_tracing and langsmith_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_PROJECT"] = langsmith_project
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- STEP 1: PROCESS & INDEX UPLOADED FILES ---
if process_button:
    if not api_key:
        st.sidebar.error("OPENAI_API_KEY is missing in your .env file.")
    elif not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF or TXT file.")
    else:
        configure_env()
        with st.spinner("Reading, chunking, and embedding your files..."):
            all_docs = []

            for uploaded_file in uploaded_files:
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    if file_ext == ".pdf":
                        loader = PyPDFLoader(tmp_path)
                    else:
                        loader = TextLoader(tmp_path, encoding="utf-8")
                    loaded = loader.load()
                    # Preserve original filename in metadata for LangSmith & citations
                    for doc in loaded:
                        doc.metadata["source"] = uploaded_file.name
                    all_docs.extend(loaded)
                finally:
                    os.remove(tmp_path)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(all_docs)

            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            st.session_state.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings
            )
            st.sidebar.success(f"Indexed {len(splits)} chunks from {len(uploaded_files)} file(s)!")

# --- STEP 2: CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a question about your uploaded files..."):
    if not api_key:
        st.error("OPENAI_API_KEY is missing in your .env file.")
    elif st.session_state.vectorstore is None:
        st.warning("Please upload and process your documents in the sidebar first.")
    else:
        configure_env()

        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer the user's question using ONLY the context provided below.
        If the answer cannot be found in the context, state that clearly.

        Context:
        {context}

        Question: {question}
        """)

        def format_docs(docs):
            return "\n\n---\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                # Invoke with run_name and tags for cleaner filtering in LangSmith
                response = rag_chain.invoke(
                    user_query,
                    config={
                        "run_name": "Streamlit_RAG_Query",
                        "tags": ["streamlit", "rag", "gpt-4o-mini"]
                    }
                )
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        