from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.meeting import Meeting
from app.models.user import User
from app.schemas.meeting import MeetingResponse
from app.core.deps import get_current_user
from typing import List

router = APIRouter()

@router.get("/", response_model=List[MeetingResponse])
def get_meetings(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    meetings = db.query(Meeting).filter(Meeting.user_id == current_user.id).all()
    return meetings

@router.get("/{meeting_id}", response_model=MeetingResponse)
def get_meeting(meeting_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id, Meeting.user_id == current_user.id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting

@router.get("/{meeting_id}/transcript")
def get_meeting_transcript(meeting_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    import os
    from app.services.meeting_library import get_meeting_dir, read_meeting_metadata
    
    meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id, Meeting.user_id == current_user.id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found in database")
        
    meeting_dir = get_meeting_dir(meeting_id)
    metadata = read_meeting_metadata(meeting_dir)
    if not metadata or not metadata.get("transcript_file"):
        raise HTTPException(status_code=404, detail="Transcript metadata not found")
        
    transcript_path = os.path.join(meeting_dir, metadata["transcript_file"])
    if not os.path.exists(transcript_path):
        raise HTTPException(status_code=404, detail="Transcript file missing on disk")
        
    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    return {"content": content}

@router.delete("/{meeting_id}")
def delete_meeting(meeting_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    import shutil
    from app.services.meeting_library import get_meeting_dir
    
    meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id, Meeting.user_id == current_user.id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
        
    # Delete from database
    db.delete(meeting)
    db.commit()
    
    # Delete from disk
    try:
        meeting_dir = get_meeting_dir(meeting_id)
        if meeting_dir.exists():
            shutil.rmtree(meeting_dir)
    except Exception as e:
        print(f"Failed to delete meeting directory: {e}")
        
    return {"message": "Meeting deleted successfully"}

@router.get("/{meeting_id}/summary")
def get_meeting_summary(meeting_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    import os
    import json
    from app.services.meeting_library import get_meeting_dir
    
    meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id, Meeting.user_id == current_user.id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found in database")
        
    meeting_dir = get_meeting_dir(meeting_id)
    summary_path = os.path.join(meeting_dir, "summary.json")
    
    if not os.path.exists(summary_path):
        return {"summary": "Summary not available yet.", "action_items": [], "key_decisions": []}
        
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)

@router.patch("/{meeting_id}/speakers")
def update_meeting_speakers(meeting_id: str, updates: dict, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    from app.services.speaker_update import update_speakers
    
    meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id, Meeting.user_id == current_user.id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found in database")
        
    try:
        update_speakers(meeting_id, updates)
        return {"message": "Speakers updated and vector store re-ingested."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{meeting_id}/analytics")
def get_meeting_analytics(meeting_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    import os
    import re
    from collections import defaultdict
    from app.services.meeting_library import get_meeting_dir, read_meeting_metadata
    
    meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id, Meeting.user_id == current_user.id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found in database")
        
    meeting_dir = get_meeting_dir(meeting_id)
    metadata = read_meeting_metadata(meeting_dir)
    if not metadata or not metadata.get("transcript_file"):
        raise HTTPException(status_code=404, detail="Transcript metadata not found")
        
    transcript_path = os.path.join(meeting_dir, metadata["transcript_file"])
    if not os.path.exists(transcript_path):
        raise HTTPException(status_code=404, detail="Transcript file missing on disk")
        
    speaker_words = defaultdict(int)
    speaker_pattern = re.compile(r"^(?:\[[^\]]+\]\s*)?([^:]+):(.*)")
    
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = speaker_pattern.match(line)
            if match:
                speaker = match.group(1).strip()
                text = match.group(2).strip()
                words = len(text.split())
                speaker_words[speaker] += words
                
    total_words = sum(speaker_words.values())
    data = []
    for spk, count in speaker_words.items():
        data.append({
            "name": spk,
            "value": count,
            "percentage": round((count / total_words) * 100) if total_words > 0 else 0
        })
        
    return {"speakerData": data, "totalWords": total_words}
