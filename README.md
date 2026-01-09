# RAG-Backed Q&A Console

This repository delivers a minimal Retrieval-Augmented Generation (RAG) workflow that lets you:

- upload documents (PDF, Markdown, plain text, JSON),
- observe the chunking + FAISS indexing progress,
- type a question into a focused box, and
- receive a grounded answer with cited sources pulled from the indexed content.

Streamlit drives the interface, LangChain + Sentence Transformers handle chunking/embeddings, and FAISS stores the vector index.

## Quick Start

1. **Install dependencies**:

   ```bash
   pip install -e .
   ```

2. **Set your OpenAI credentials** (needed for `ChatOpenAI`):

   ```bash
   export OPENAI_API_KEY=sk-...
   ```

3. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

4. **Follow the UI steps**:

   - Drag / drop or select source files in the upload panel.
   - Click **Build FAISS index** to chunk, embed, and persist the vector store.
   - Ask a question in the text box and hit **Search & Generate**.
   - Review the generated answer plus the referenced snippets from your documents.

## Features implemented

| Step | Behavior |
| --- | --- |
| Upload | Displays file count/size and stores uploads in a temporary workspace. |
| Processing | Streams progress updates while saving files, loading docs, chunking, and building a FAISS index. |
| Question | Provides a dedicated area to enter questions once the index is ready. |
| Retrieval | Runs `ChatOpenAI` over retrieved chunks via LangChain's `RetrievalQA`. |
| Display | Shows the generated answer plus metadata-backed source snippets. |

## Notes

- Unsupported file types still appear in the workspace with a metadata note, even if the content is empty.
- The temporary workspace survives per Streamlit session to make debugging or inspection easier.
- FAISS persistence happens in the temporary workspace under `faiss_index/`.
- This setup assumes access to OpenAI; swap in a different Chat model if you need a local inference stack.

