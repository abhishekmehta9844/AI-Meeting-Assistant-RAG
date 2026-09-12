from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.meeting import Meeting
from app.services.meeting_bot import MeetingBot
from pydantic import BaseModel
import uuid
import datetime
import threading

router = APIRouter()

# Global dictionary to track active bots by meeting_id
active_bots = {}

class BotStartRequest(BaseModel):
    url: str
    title: str = "New Meeting"
    transcriber: str = "whisper"

def run_bot_task(meeting_id: str, url: str, transcriber: str, db: Session):
    print(f"Starting bot for {url} with ID {meeting_id}")
    try:
        bot = MeetingBot(url, transcription_engine=transcriber, meeting_id=meeting_id)
        
        # Track the bot in memory
        active_bots[meeting_id] = bot
        
        # Run the bot. This blocks until stop_recording is set.
        bot.run()
        
        # When bot.run() finishes, processing and ingestion are complete.
        print(f"Bot {meeting_id} finished execution and processing.")
        
        # Update database status
        meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id).first()
        if meeting:
            meeting.status = "completed"
            db.commit()
    except Exception as e:
        print(f"Bot error: {e}")
        meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id).first()
        if meeting:
            meeting.status = "failed"
            db.commit()
    finally:
        # Cleanup
        if meeting_id in active_bots:
            del active_bots[meeting_id]

from app.core.deps import get_current_user
from app.models.user import User

@router.post("/start")
def start_bot(req: BotStartRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    unique_id = str(uuid.uuid4())
    
    new_meeting = Meeting(
        meeting_id=unique_id,
        title=req.title,
        url=req.url,
        started_at=datetime.datetime.utcnow(),
        status="recording",
        user_id=current_user.id
    )
    db.add(new_meeting)
    db.commit()
    db.refresh(new_meeting)

    background_tasks.add_task(run_bot_task, unique_id, req.url, req.transcriber, db)

    return {"message": "Bot triggered successfully", "meeting_id": unique_id}

@router.post("/{meeting_id}/stop")
def stop_bot(meeting_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id, Meeting.user_id == current_user.id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found or you don't have access")
        
    bot = active_bots.get(meeting_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not active")
    
    # Signal the bot to stop recording and proceed to transcription
    bot.stop_recording.set()
    return {"message": "Bot stop signaled. Transcription starting."}
