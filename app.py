import os
import tempfile
import uuid
from pathlib import Path
from typing import Iterable, List, Tuple

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-3.5-turbo"


def create_workspace() -> Path:
    root = Path(tempfile.gettempdir()) / "rag_streamlit_workspace"
    root.mkdir(exist_ok=True, parents=True)
    return root / uuid.uuid4().hex


def save_uploaded_files(files: Iterable[st.uploaded_file_manager.UploadedFile], destination: Path) -> List[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    for upload in files:
        target = destination / f"{uuid.uuid4().hex}_{upload.name}"
        with open(target, "wb") as buffer:
            buffer.write(upload.getbuffer())
        saved_paths.append(target)
    return saved_paths


def load_documents(paths: Iterable[Path]) -> List[Document]:
    docs: List[Document] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".log", ".json"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": path.name, "type": "text"}))
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            docs.extend(loader.load())
        else:
            docs.append(
                Document(
                    page_content="",
                    metadata={"source": path.name, "note": f"file type {suffix} is not parsed automatically"},
                )
            )
    return docs


def build_faiss_index(documents: List[Document]) -> Tuple[FAISS, int, SentenceTransformerEmbeddings]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embedding = SentenceTransformerEmbeddings(model_name=MODEL_NAME)
    store = FAISS.from_documents(chunks, embedding)
    return store, len(chunks), embedding


def create_qa_chain(vector_store: FAISS) -> RetrievalQA:
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0.1)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )


def main() -> None:
    st.set_page_config(page_title="RAG Q&A Console", layout="wide")
    st.title("Retrieval-Augmented Generation: Q&A Console")
    st.caption("Upload files, inspect the indexing progress, and ask grounded questions.")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "analysis" not in st.session_state:
        st.session_state.analysis = {}
    if "workspace_root" not in st.session_state:
        st.session_state.workspace_root = None

    instructions = (
        "1. Upload documents (PDF, TXT, Markdown, JSON).  "
        "2. Process them to chunk, embed, and build a FAISS index.  "
        "3. Ask a question in the box below.  "
        "4. Receive an answer with cited sources."
    )
    st.info(instructions)

    upload_col, process_col = st.columns(2)

    with upload_col:
        st.subheader("1. Upload files")
        uploaded_files = st.file_uploader(
            "Drag & drop files or select them manually",
            type=["txt", "md", "pdf", "json"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.success(f"{len(uploaded_files)} file(s) staged for processing")
            for upload in uploaded_files:
                st.caption(f"- {upload.name} ({upload.size / 1024:.1f} KB)")
        elif st.session_state.get("uploaded_files"):
            st.caption("Use the same uploads or add more files before processing.")

    with process_col:
        st.subheader("2. Process data")
        step_placeholder = st.empty()
        progress_bar = st.progress(0)
        process_trigger = st.button("Build FAISS index")
        if process_trigger:
            if not st.session_state.get("uploaded_files"):
                st.warning("Upload at least one file before building the index.")
            else:
                workspace = create_workspace()
                st.session_state.workspace_root = workspace
                step_placeholder.info("Copying uploads to workspace…")
                progress_bar.progress(10)
                saved_paths = save_uploaded_files(st.session_state.uploaded_files, workspace)
                step_placeholder.info("Loading documents")
                progress_bar.progress(35)
                documents = load_documents(saved_paths)
                if not documents:
                    step_placeholder.error("No parseable content found.")
                    progress_bar.progress(100)
                else:
                    step_placeholder.info("Chunking documents & training FAISS index")
                    store, chunk_count, embedding = build_faiss_index(documents)
                    index_location = workspace / "faiss_index"
                    store.save_local(str(index_location))
                    st.session_state.vector_store = store
                    st.session_state.analysis = {
                        "documents": len(documents),
                        "chunks": chunk_count,
                        "workspace": str(workspace),
                        "embed_model": MODEL_NAME,
                    }
                    step_placeholder.success("Vector index ready for retrieval.")
                    progress_bar.progress(100)
        elif st.session_state.vector_store:
            step_placeholder.success("FAISS index already available.")
            progress_bar.progress(100)

    if st.session_state.analysis:
        st.markdown("#### Index digestion stats")
        stats = st.session_state.analysis
        st.write(
            f"- Sources indexed: {stats['documents']}\n"
            f"- Total chunks: {stats['chunks']}\n"
            f"- Embedding model: {stats['embed_model']}\n"
            f"- Workspace: `{stats['workspace']}`"
        )

    st.markdown("---")
    st.subheader("3. Ask a question")
    query = st.text_area("Type your question here", height=140)
    ask_button = st.button("Search & Generate")

    if ask_button:
        if not st.session_state.vector_store:
            st.warning("Process files to build the index before asking a question.")
        elif not query.strip():
            st.warning("Enter a question before submitting.")
        else:
            try:
                qa_chain = create_qa_chain(st.session_state.vector_store)
                with st.spinner("Searching the index and generating an answer…"):
                    result = qa_chain({"query": query})
                    st.session_state.last_result = result
            except Exception as exc:
                st.error(f"LLM inference failed: {exc}")

    if st.session_state.get("last_result"):
        answer = st.session_state.last_result["result"]
        sources = st.session_state.last_result.get("source_documents", [])
        st.markdown("#### Answer")
        st.write(answer)
        st.markdown("#### Source documents and snippets")
        for idx, doc in enumerate(sources, start=1):
            source = doc.metadata.get("source", f"document-{idx}")
            snippet = doc.page_content.strip().replace("\n", " ")[:400]
            st.write(f"**{idx}. {source}** – {snippet}{'...' if len(snippet) == 400 else ''}")
            if metadata := {k: v for k, v in doc.metadata.items() if k not in {"source"}}:
                st.caption(f"Metadata: {metadata}")

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("Set the `OPENAI_API_KEY` environment variable so the ChatOpenAI client can run.")


if __name__ == "__main__":
    main()

