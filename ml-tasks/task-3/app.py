import os
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv


from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GOOGLE_API_MODEL = os.getenv("GOOGLE_API_MODEL", "gemini-1.5-mini")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


def pdf_to_docs(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    docs = splitter.split_documents(pages)
    return docs


def build_vectorstore(docs: List[Document], persist_dir: str = CHROMA_PERSIST_DIR) -> Chroma:
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    vectordb.persist()
    return vectordb


def retrieve_and_answer(vectordb: Chroma, question: str, k: int = 4) -> str:
    results = vectordb.similarity_search(query=question, k=k)
    context_text = "\n\n---\n\n".join([d.page_content for d in results])

    prompt_template = """You are a helpful assistant that must answer questions using the provided context. talk in natural language not just one word answers,
If the answer is not contained in the context, say: "I don't know — the document does not contain that information."

Context:
{context}

Question:
{question}

Answer concisely, cite which chunk you used by prefixing the answer with [source N] where N is the chunk index (1-based) from the top-k results. If you used multiple chunks, list them like [source 1,3].
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )


    llm = ChatGoogleGenerativeAI(model=GOOGLE_API_MODEL, temperature=0.0)

    chain = LLMChain(llm=llm, prompt=prompt)
    out = chain.run({"context": context_text, "question": question})
    return out

st.set_page_config(page_title="PDF Q&A (Gemini + Chroma)", layout="wide")
st.title("PDF Q&A — Gemini + Chroma (LangChain)")

st.markdown(
    """
Upload a PDF, build an index (stored in Chroma), then ask questions.  
- Paste your Gemini API key into the `.env` as `GOOGLE_API_KEY`.  
- Embeddings use a local SentenceTransformer model (no embedding API key required).
"""
)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    save_path = Path("uploads")
    save_path.mkdir(exist_ok=True)
    file_location = save_path / uploaded_file.name
    with open(file_location, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved file to {file_location}")

    if st.button("Build index from this PDF"):
        with st.spinner("Extracting text and building vector index (this may take a minute)..."):
            docs = pdf_to_docs(str(file_location))
            vectordb = build_vectorstore(docs)
        st.success("Index built and persisted to Chroma.")
        st.session_state["vectordb_present"] = True

if "vectordb_present" not in st.session_state:
    if Path(CHROMA_PERSIST_DIR).exists() and any(Path(CHROMA_PERSIST_DIR).iterdir()):
        st.info(f"Found existing Chroma DB in {CHROMA_PERSIST_DIR}. You can ask questions against it.")
        st.session_state["vectordb_present"] = True

if st.session_state.get("vectordb_present"):
    st.markdown("---")
    st.subheader("Ask a question from the indexed document(s)")
    question = st.text_input("Your question:")
    top_k = st.slider("Top K chunks to retrieve", min_value=1, max_value=8, value=4)
    if st.button("Get answer") and question.strip():
        with st.spinner("Searching and calling Gemini..."):
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
            answer = retrieve_and_answer(vectordb, question, k=top_k)
        st.markdown("**Answer:**")
        st.write(answer)
