from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class MeetingBase(BaseModel):
    title: str
    url: str

class MeetingCreate(MeetingBase):
    pass

class MeetingResponse(MeetingBase):
    id: int
    meeting_id: str
    started_at: datetime
    status: str
    audio_file_path: Optional[str] = None
    transcript_file_path: Optional[str] = None

    class Config:
        from_attributes = True

