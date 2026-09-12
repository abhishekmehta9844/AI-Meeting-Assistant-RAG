import sys
import os
import json
from pathlib import Path
from sqlalchemy.orm import Session
from app.core.database import SessionLocal, Base, engine
from app.models.meeting import Meeting
import datetime

def sync_meetings():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    meetings_dir = Path("../data/meetings")
    if not meetings_dir.exists():
        print("Data directory not found.")
        return

    for meeting_path in meetings_dir.iterdir():
        if not meeting_path.is_dir():
            continue
        
        metadata_file = meeting_path / "metadata.json"
        if not metadata_file.exists():
            continue
            
        try:
            with open(metadata_file, "r") as f:
                data = json.load(f)
                
            # Check if it already exists
            existing = db.query(Meeting).filter(Meeting.meeting_id == data["meeting_id"]).first()
            if not existing:
                meeting = Meeting(
                    meeting_id=data["meeting_id"],
                    title=data.get("title", "Unknown Meeting"),
                    url=data.get("meeting_url", ""),
                    started_at=datetime.datetime.fromisoformat(data["started_at"]) if "started_at" in data else datetime.datetime.utcnow(),
                    status="completed",
                    audio_file_path=data.get("audio_file"),
                    transcript_file_path=data.get("transcript_file"),
                    vector_store_id=data.get("collection_name")
                )
                db.add(meeting)
                print(f"Added meeting: {meeting.title}")
        except Exception as e:
            print(f"Error processing {meeting_path.name}: {e}")
            
    db.commit()
    db.close()
    print("Sync complete.")

if __name__ == "__main__":
    sync_meetings()

