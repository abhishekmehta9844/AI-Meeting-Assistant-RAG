from sqlalchemy import Column, Integer, String, DateTime
from app.core.database import Base
import datetime

class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(String, unique=True, index=True)
    title = Column(String)
    url = Column(String)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default="recording") # recording, transcribing, indexing, completed, failed
    audio_file_path = Column(String, nullable=True)
    transcript_file_path = Column(String, nullable=True)
    vector_store_id = Column(String, nullable=True)
    user_id = Column(Integer, index=True, nullable=True) # Optional for backward compatibility with existing meetings

