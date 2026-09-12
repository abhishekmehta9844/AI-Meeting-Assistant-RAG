import os
import json
import re
from app.services.meeting_library import get_meeting_dir, read_meeting_metadata, write_meeting_metadata
from app.services.ingest import ingest

def update_speakers(meeting_id: str, updates: dict):
    meeting_dir = get_meeting_dir(meeting_id)
    metadata = read_meeting_metadata(meeting_dir)
    
    if not metadata:
        raise FileNotFoundError("Meeting metadata not found")
        
    transcript_file = os.path.join(meeting_dir, metadata.get("transcript_file", ""))
    document_json = os.path.join(meeting_dir, metadata.get("document_json", ""))
    
    if not os.path.exists(transcript_file) or not os.path.exists(document_json):
        raise FileNotFoundError("Transcript or document JSON not found")
        
    # Update transcript file
    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript_content = f.read()
        
    for old_spk, new_spk in updates.items():
        # Match "SPEAKER_0:" or "[00:00 - 00:01] SPEAKER_0:"
        transcript_content = re.sub(rf'\b{re.escape(old_spk)}\b', new_spk, transcript_content, flags=re.IGNORECASE)
        
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript_content)
        
    # Update document json
    with open(document_json, "r", encoding="utf-8") as f:
        doc_data = json.load(f)
        
    doc_content = doc_data.get("content", "")
    doc_participants = doc_data.get("participants", [])
    
    for old_spk, new_spk in updates.items():
        doc_content = re.sub(rf'\b{re.escape(old_spk)}\b', new_spk, doc_content, flags=re.IGNORECASE)
        # Case insensitive remove from participants
        doc_participants = [p for p in doc_participants if p.lower() != old_spk.lower()]
        if new_spk not in doc_participants:
            doc_participants.append(new_spk)
                
    doc_data["content"] = doc_content
    doc_data["participants"] = doc_participants
    
    with open(document_json, "w", encoding="utf-8") as f:
        json.dump(doc_data, f, indent=4)
        
    # Update metadata
    meta_participants = metadata.get("participants", [])
    meta_preview = metadata.get("transcript_preview", "")
    for old_spk, new_spk in updates.items():
        meta_participants = [p for p in meta_participants if p.lower() != old_spk.lower()]
        if new_spk not in meta_participants:
            meta_participants.append(new_spk)
        meta_preview = re.sub(rf'\b{re.escape(old_spk)}\b', new_spk, meta_preview, flags=re.IGNORECASE)
        
    metadata["participants"] = meta_participants
    metadata["transcript_preview"] = meta_preview
    write_meeting_metadata(meeting_dir, metadata)
    
    # Re-ingest
    vector_store_dir = meeting_dir / "vector_store"
    collection_name = metadata.get("collection_name", f"meeting_{meeting_id}")
    
    ingest(
        pdf_dir=None,
        json_dir=str(meeting_dir / "documents"),
        persist_dir=str(vector_store_dir),
        collection_name=collection_name,
        reset_collection=True
    )
    
    # Re-ingest into global as well
    global_dir = str(vector_store_dir.parent.parent / "global_vector_store")
    # We don't reset the global collection, just let it append. 
    # Note: To be perfectly clean, we'd need to delete the old chunks from global DB first.
    # But for a simple MVP, appending is okay.
    from app.services.rag_pipeline import VectorStore, EmbeddingManager, process_all_jsons, split_documents
    try:
        global_store = VectorStore(collection_name="global_meetings", persistent_directory=global_dir)
        jsons = process_all_jsons(str(meeting_dir / "documents"))
        if jsons:
            chunks = split_documents(jsons)
            embedder = EmbeddingManager()
            texts = [c.page_content for c in chunks]
            embeddings = embedder.generate_embeddings(texts)
            global_store.add_documents(chunks, embeddings)
    except Exception as e:
        print(f"Failed to update global collection: {e}")

