import json
import re
from datetime import datetime
from pathlib import Path


import os
MEETINGS_ROOT = Path(os.path.abspath(__file__)).parent.parent.parent.parent / "data" / "meetings"


def ensure_meetings_root() -> Path:
    MEETINGS_ROOT.mkdir(parents=True, exist_ok=True)
    return MEETINGS_ROOT


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return value or "meeting"


def make_meeting_id(started_at: datetime | None = None, label: str | None = None) -> str:
    started_at = started_at or datetime.now()
    suffix = slugify(label) if label else "session"
    return f"{started_at.strftime('%Y%m%d_%H%M%S')}_{suffix}"


def get_meeting_dir(meeting_id: str) -> Path:
    return ensure_meetings_root() / meeting_id


def read_meeting_metadata(meeting_dir: Path) -> dict | None:
    metadata_path = meeting_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_meeting_metadata(meeting_dir: Path, metadata: dict) -> Path:
    meeting_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = meeting_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")
    return metadata_path


def list_meetings() -> list[dict]:
    root = ensure_meetings_root()
    meetings = []
    for meeting_dir in root.iterdir():
        if not meeting_dir.is_dir():
            continue
        metadata = read_meeting_metadata(meeting_dir)
        if not metadata:
            continue
        metadata["meeting_id"] = metadata.get("meeting_id", meeting_dir.name)
        metadata["meeting_dir"] = str(meeting_dir)
        meetings.append(metadata)

    def sort_key(item: dict):
        return item.get("started_at", "")

    meetings.sort(key=sort_key, reverse=True)
    return meetings
