# ingest.py
from app.services.rag_pipeline import (
    process_all_pdfs,
    process_all_jsons,
    split_documents,
    EmbeddingManager,
    VectorStore
)

def ingest(
    pdf_dir="./data/pdfs",
    json_dir="./data/jsons",
    persist_dir="./data/vector_store",
    collection_name="knowledge_base",
    reset_collection=False,
):
    print("🚀 Starting ingestion")

    pdfs = process_all_pdfs(pdf_dir) if pdf_dir else []
    jsons = process_all_jsons(json_dir) if json_dir else []
    documents = pdfs + jsons

    if not documents:
        raise RuntimeError("No documents found")

    chunks = split_documents(documents)

    embedder = EmbeddingManager()
    texts = [c.page_content for c in chunks]
    embeddings = embedder.generate_embeddings(texts)

    store = VectorStore(
        collection_name=collection_name,
        persistent_directory=persist_dir
    )
    if reset_collection:
        store.reset_collection()
    store.add_documents(chunks, embeddings)
    
    # Also ingest into global store
    from pathlib import Path
    global_dir = str(Path(persist_dir).parent.parent / "global_vector_store")
    global_store = VectorStore(
        collection_name="global_meetings",
        persistent_directory=global_dir
    )
    # We do NOT reset the global collection!
    global_store.add_documents(chunks, embeddings)

    print(" Ingestion complete")

if __name__ == "__main__":
    ingest()
