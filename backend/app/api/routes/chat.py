from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
import os
import json
from app.core.database import get_db
from app.models.meeting import Meeting
from app.services.rag_pipeline import EmbeddingManager, VectorStore, RAGRetriever, rag_advanced_v2_stream
from langchain_groq import ChatGroq
from dotenv import load_dotenv

router = APIRouter()
load_dotenv()

class ChatRequest(BaseModel):
    meeting_id: str
    query: str
    retrieve_k: int = 12
    final_k: int = 4
    min_score: float = 0.25

# Cache for loaded vector stores to avoid re-initializing ChromaDB client
vector_stores_cache = {}
embedder = EmbeddingManager()
llm = ChatGroq(
    groq_api_key=os.environ.get("GROQ_API_KEY", ""),
    model="openai/gpt-oss-120b",
    temperature=0.1,
    max_tokens=1024
)

from app.core.deps import get_current_user
from app.models.user import User

@router.post("/")
def chat_with_meeting(req: ChatRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if req.meeting_id == "global":
        from pathlib import Path
        vector_store_dir = str(Path("./data/global_vector_store").resolve())
        collection_name = "global_meetings"
    else:
        meeting = db.query(Meeting).filter(Meeting.meeting_id == req.meeting_id, Meeting.user_id == current_user.id).first()
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found or you don't have access")

        from app.services.meeting_library import get_meeting_dir, read_meeting_metadata
        
        meeting_dir_path = get_meeting_dir(req.meeting_id)
        metadata = read_meeting_metadata(meeting_dir_path)
        
        vector_store_dir = str(meeting_dir_path / "vector_store")
        collection_name = meeting.vector_store_id
        
        if metadata:
            collection_name = metadata.get("collection_name") or collection_name
            
        if not collection_name:
            collection_name = f"meeting_{req.meeting_id.split('_')[0]}_{req.meeting_id.split('_')[1]}"

    if collection_name not in vector_stores_cache:
        try:
            vector_stores_cache[collection_name] = VectorStore(
                collection_name=collection_name,
                persistent_directory=vector_store_dir
            )
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise HTTPException(status_code=500, detail="Failed to load meeting context")

    store = vector_stores_cache[collection_name]
    retriever = RAGRetriever(store, embedder)

    def generate():
        try:
            for item in rag_advanced_v2_stream(
                query=req.query,
                retriever=retriever,
                llm=llm,
                retrieve_k=req.retrieve_k,
                final_k=req.final_k,
                min_score=req.min_score
            ):
                yield f"data: {json.dumps(item)}\n\n"
        except Exception as e:
            error_msg = f'\n\n[Error: {str(e)}]'
            yield f"data: {json.dumps({'type': 'token', 'content': error_msg })}\n\n"
        finally:
            yield "data: [DONE]\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream")
