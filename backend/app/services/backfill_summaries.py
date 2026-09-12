import os
import json
import traceback
from app.core.database import SessionLocal
from app.models.meeting import Meeting
from app.services.meeting_library import get_meeting_dir, read_meeting_metadata
from app.services.summary import generate_meeting_summary

def main():
    db = SessionLocal()
    meetings = db.query(Meeting).all()
    
    for meeting in meetings:
        meeting_dir = get_meeting_dir(meeting.meeting_id)
        summary_path = os.path.join(meeting_dir, "summary.json")
        
        if os.path.exists(summary_path):
            print(f"Skipping {meeting.meeting_id} - summary already exists.")
            continue
            
        metadata = read_meeting_metadata(meeting_dir)
        if not metadata or not metadata.get("transcript_file"):
            print(f"Skipping {meeting.meeting_id} - no metadata/transcript file.")
            continue
            
        transcript_path = os.path.join(meeting_dir, metadata["transcript_file"])
        if not os.path.exists(transcript_path):
            print(f"Skipping {meeting.meeting_id} - transcript file missing.")
            continue
            
        print(f"Generating summary for {meeting.meeting_id}...")
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            summary_data = generate_meeting_summary(content)
            
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=4)
            print(f"  -> Saved summary!")
        except Exception as e:
            print(f"  -> Failed: {e}")
            
    print("Done!")

if __name__ == "__main__":
    main()
