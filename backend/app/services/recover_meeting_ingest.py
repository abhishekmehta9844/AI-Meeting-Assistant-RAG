from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.services.ingest import ingest
from app.services.meeting_library import write_meeting_metadata
from app.services.rag_pipeline import build_rag_v2


MEETINGS_ROOT = Path("./data/meetings")
SUPPORTED_MEDIA_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".flac",
    ".ogg",
    ".aac",
    ".wma",
    ".avi",
}
SPEAKER_PATTERN = re.compile(r"^(?:\[[^\]]+\]\s*)?([^:]+):")


@dataclass
class MeetingCandidate:
    meeting_dir: Path
    meeting_id: str
    metadata: dict
    audio_files: list[Path]
    transcript_path: Path | None
    document_json_path: Path | None
    vector_store_dir: Path
    collection_name: str


def load_metadata(meeting_dir: Path) -> dict:
    metadata_path = meeting_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_audio_files(meeting_dir: Path) -> list[Path]:
    audio_files = [
        path
        for path in meeting_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS
    ]
    audio_files.sort(key=lambda path: (path.stat().st_size, path.name), reverse=True)
    return audio_files


def resolve_document_json_path(meeting_dir: Path, metadata: dict) -> Path | None:
    raw_path = metadata.get("document_json")
    if raw_path:
        candidate = meeting_dir / Path(raw_path)
        if candidate.exists():
            return candidate

    documents_dir = meeting_dir / "documents"
    if not documents_dir.exists():
        return None

    json_files = sorted(documents_dir.glob("*.json"))
    return json_files[0] if json_files else None


def transcript_name_for(audio_file: Path) -> str:
    return f"{audio_file.stem}_transcript.txt"


def move_into_meeting_dir(source_path: Path, meeting_dir: Path, target_name: str) -> Path:
    destination = meeting_dir / target_name
    if source_path.resolve() == destination.resolve():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_path.replace(destination)
    return destination


def reconstruct_transcript_from_json(document_json_path: Path, transcript_path: Path) -> Path | None:
    try:
        payload = json.loads(document_json_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    content = str(payload.get("content", "")).strip()
    if not content:
        return None

    transcript_path.write_text(content + "\n", encoding="utf-8")
    return transcript_path


def find_existing_transcript(meeting_dir: Path, metadata: dict, audio_files: list[Path]) -> Path | None:
    candidate_names = []

    for audio_file in audio_files:
        candidate_names.append(transcript_name_for(audio_file))

    metadata_transcript = metadata.get("transcript_file")
    if metadata_transcript:
        candidate_names.append(Path(metadata_transcript).name)

    seen = set()
    for candidate_name in candidate_names:
        if candidate_name in seen:
            continue
        seen.add(candidate_name)

        in_meeting_dir = meeting_dir / candidate_name
        if in_meeting_dir.exists():
            return in_meeting_dir

        in_cwd = Path.cwd() / candidate_name
        if in_cwd.exists():
            return move_into_meeting_dir(in_cwd, meeting_dir, candidate_name)

    document_json_path = resolve_document_json_path(meeting_dir, metadata)
    if document_json_path:
        transcript_file_name = transcript_name_for(audio_files[0]) if audio_files else f"{meeting_dir.name}_transcript.txt"
        transcript_path = meeting_dir / transcript_file_name
        rebuilt = reconstruct_transcript_from_json(document_json_path, transcript_path)
        if rebuilt:
            return rebuilt

    return None


def infer_started_at(meeting_id: str, metadata: dict) -> datetime:
    started_at = metadata.get("started_at")
    if started_at:
        try:
            return datetime.fromisoformat(started_at)
        except ValueError:
            pass

    try:
        return datetime.strptime(meeting_id[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return datetime.now()


def infer_collection_name(meeting_id: str, metadata: dict) -> str:
    if metadata.get("collection_name"):
        return str(metadata["collection_name"])
    return f"meeting_{meeting_id[:15]}"


def discover_meetings() -> list[MeetingCandidate]:
    if not MEETINGS_ROOT.exists():
        return []

    meetings: list[MeetingCandidate] = []
    for meeting_dir in sorted(MEETINGS_ROOT.iterdir(), reverse=True):
        if not meeting_dir.is_dir():
            continue

        metadata = load_metadata(meeting_dir)
        audio_files = find_audio_files(meeting_dir)
        document_json_path = resolve_document_json_path(meeting_dir, metadata)
        transcript_path = find_existing_transcript(meeting_dir, metadata, audio_files)
        collection_name = infer_collection_name(meeting_dir.name, metadata)

        if not audio_files and not document_json_path:
            continue

        meetings.append(
            MeetingCandidate(
                meeting_dir=meeting_dir,
                meeting_id=meeting_dir.name,
                metadata=metadata,
                audio_files=audio_files,
                transcript_path=transcript_path,
                document_json_path=document_json_path,
                vector_store_dir=meeting_dir / "vector_store",
                collection_name=collection_name,
            )
        )

    return meetings


def format_meeting_label(meeting: MeetingCandidate) -> str:
    started_at = infer_started_at(meeting.meeting_id, meeting.metadata)
    title = meeting.metadata.get("title") or meeting.meeting_id
    audio_status = "audio" if meeting.audio_files else "no-audio"
    transcript_status = "transcript" if meeting.transcript_path else "no-transcript"
    json_status = "json" if meeting.document_json_path else "no-json"
    return (
        f"{title} | {started_at.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"{audio_status}, {transcript_status}, {json_status}"
    )


def choose_meeting(meetings: list[MeetingCandidate], explicit_value: str | None) -> MeetingCandidate:
    if explicit_value:
        normalized = explicit_value.strip()
        if normalized.isdigit():
            index = int(normalized) - 1
            if 0 <= index < len(meetings):
                return meetings[index]
        for meeting in meetings:
            if meeting.meeting_id == normalized:
                return meeting
        raise ValueError(f"No meeting matched '{explicit_value}'.")

    print("\nSaved meetings:\n")
    for index, meeting in enumerate(meetings, start=1):
        print(f"{index}. {format_meeting_label(meeting)}")

    while True:
        choice = input("\nChoose a meeting number: ").strip()
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(meetings):
                return meetings[index]
        print("Please enter one valid meeting number.")


def choose_transcriber(explicit_choice: str | None, metadata: dict) -> str:
    if explicit_choice in {"whisper", "sarvam"}:
        return explicit_choice

    default_choice = metadata.get("transcription_engine")
    if default_choice not in {"whisper", "sarvam"}:
        default_choice = "sarvam"

    print("\nChoose transcription engine:")
    print("1. sarvam")
    print("2. whisper")
    prompt = "Enter 1 or 2"
    prompt += f" [default {1 if default_choice == 'sarvam' else 2}]: "

    while True:
        choice = input(prompt).strip().lower()
        if choice == "":
            return default_choice
        if choice in {"1", "sarvam"}:
            return "sarvam"
        if choice in {"2", "whisper"}:
            return "whisper"
        print("Please enter 1 for sarvam or 2 for whisper.")


def prompt_yes_no(question: str, default: bool = True) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        value = input(question + suffix).strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def extract_participants(transcript_text: str) -> list[str]:
    participants = []
    seen = set()

    for raw_line in transcript_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = SPEAKER_PATTERN.match(line)
        if not match:
            continue

        speaker = match.group(1).strip()
        normalized = speaker.lower()
        if not speaker or normalized in seen:
            continue

        seen.add(normalized)
        participants.append(speaker)

    return participants


def transcribe_audio(audio_file: Path, transcriber: str, meeting_dir: Path) -> tuple[Path, str]:
    transcript_name = transcript_name_for(audio_file)

    if transcriber == "sarvam":
        from app.services import sarvam_transcribe

        generated_path = Path(sarvam_transcribe.process_single_file(str(audio_file)))
    elif transcriber == "whisper":
        from app.services import cloudtranscribe

        cloudtranscribe.process_single_file(str(audio_file))
        generated_path = Path(cloudtranscribe.make_output_name(str(audio_file)))
    else:
        raise ValueError(f"Unsupported transcriber '{transcriber}'.")

    if not generated_path.is_absolute():
        generated_path = Path.cwd() / generated_path
    if not generated_path.exists():
        raise FileNotFoundError(f"Transcript was not generated for {audio_file.name}.")

    final_transcript_path = move_into_meeting_dir(generated_path, meeting_dir, transcript_name)
    transcript_text = final_transcript_path.read_text(encoding="utf-8")
    return final_transcript_path, transcript_text


def ensure_transcript(
    meeting: MeetingCandidate,
    *,
    explicit_transcriber: str | None,
    force_transcribe: bool,
) -> tuple[Path, str, str]:
    if meeting.transcript_path and not force_transcribe:
        transcript_text = meeting.transcript_path.read_text(encoding="utf-8")
        transcriber = str(meeting.metadata.get("transcription_engine") or explicit_transcriber or "sarvam")
        return meeting.transcript_path, transcript_text, transcriber

    if not meeting.audio_files:
        if meeting.document_json_path:
            transcript_path = meeting.meeting_dir / f"{meeting.meeting_id}_transcript.txt"
            rebuilt = reconstruct_transcript_from_json(meeting.document_json_path, transcript_path)
            if rebuilt:
                transcript_text = rebuilt.read_text(encoding="utf-8")
                transcriber = str(meeting.metadata.get("transcription_engine") or explicit_transcriber or "sarvam")
                return rebuilt, transcript_text, transcriber
        raise RuntimeError("No audio file is available for this meeting.")

    transcriber = choose_transcriber(explicit_transcriber, meeting.metadata)
    print(f"\nRunning transcription with {transcriber} for {meeting.audio_files[0].name} ...\n")
    transcript_path, transcript_text = transcribe_audio(meeting.audio_files[0], transcriber, meeting.meeting_dir)
    meeting.transcript_path = transcript_path
    return transcript_path, transcript_text, transcriber


def build_payload(
    meeting: MeetingCandidate,
    transcript_path: Path,
    transcript_text: str,
    transcriber: str,
) -> tuple[dict, Path]:
    started_at = infer_started_at(meeting.meeting_id, meeting.metadata)
    documents_dir = meeting.meeting_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    audio_name = meeting.audio_files[0].name if meeting.audio_files else meeting.metadata.get("audio_file", "")
    participants = extract_participants(transcript_text)
    json_path = documents_dir / f"{transcript_path.stem}.json"

    payload = {
        "title": meeting.metadata.get("title") or f"Meeting {started_at.strftime('%Y-%m-%d %H:%M')}",
        "content": transcript_text,
        "source_audio": audio_name,
        "source_transcript": transcript_path.name,
        "type": "meeting_transcript",
        "meeting_id": meeting.meeting_id,
        "meeting_url": meeting.metadata.get("meeting_url", ""),
        "started_at": started_at.isoformat(),
        "participants": participants,
        "transcription_engine": transcriber,
    }

    json_path.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")
    meeting.document_json_path = json_path
    return payload, json_path


def update_metadata(
    meeting: MeetingCandidate,
    payload: dict,
    transcript_path: Path,
    document_json_path: Path,
) -> None:
    metadata = dict(meeting.metadata)
    metadata.update(
        {
            "meeting_id": meeting.meeting_id,
            "title": payload["title"],
            "meeting_url": payload["meeting_url"],
            "started_at": payload["started_at"],
            "audio_file": payload["source_audio"],
            "transcript_file": transcript_path.name,
            "document_json": str(document_json_path.relative_to(meeting.meeting_dir)),
            "vector_store_dir": str(meeting.vector_store_dir),
            "collection_name": meeting.collection_name,
            "transcript_preview": payload["content"][:300],
            "participants": payload["participants"],
            "transcription_engine": payload["transcription_engine"],
        }
    )
    write_meeting_metadata(meeting.meeting_dir, metadata)
    meeting.metadata = metadata


def ingest_meeting(meeting: MeetingCandidate) -> None:
    print("\nStarting ingestion for this meeting only...\n")
    ingest(
        pdf_dir=None,
        json_dir=str(meeting.meeting_dir / "documents"),
        persist_dir=str(meeting.vector_store_dir),
        collection_name=meeting.collection_name,
        reset_collection=True,
    )


def chat_with_meeting(meeting: MeetingCandidate) -> None:
    print("\nLoading meeting chat...\n")
    ask = build_rag_v2(
        persist_dir=str(meeting.vector_store_dir),
        collection_name=meeting.collection_name,
    )

    print("Chat is ready. Type your question.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Chat closed.")
            return

        result = ask(question, retrieve_k=8, final_k=4, min_score=0.2)
        print(f"\nAssistant: {result.get('answer', 'I do not know.')}\n")

        sources = result.get("sources") or []
        if sources:
            print("Sources:")
            for source in sources:
                print(f"- {source.get('source', 'unknown')} (score: {source.get('score', 0)})")
            print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pick one saved meeting, rebuild its transcript/json/vector store, and optionally chat with it."
    )
    parser.add_argument(
        "--meeting",
        help="Meeting number from the list or exact meeting folder name.",
    )
    parser.add_argument(
        "--transcriber",
        choices=["sarvam", "whisper"],
        help="Force a transcription engine when re-transcribing.",
    )
    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Re-run transcription even if a transcript or document JSON already exists.",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Skip the terminal chat after ingestion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meetings = discover_meetings()
    if not meetings:
        raise RuntimeError(f"No meetings found in {MEETINGS_ROOT}.")

    meeting = choose_meeting(meetings, args.meeting)
    print(f"\nSelected meeting: {meeting.meeting_id}")

    force_transcribe = args.force_transcribe
    if meeting.transcript_path and not force_transcribe:
        force_transcribe = prompt_yes_no("Transcript already exists. Re-transcribe from audio?", default=False)

    transcript_path, transcript_text, transcriber = ensure_transcript(
        meeting,
        explicit_transcriber=args.transcriber,
        force_transcribe=force_transcribe,
    )

    payload, document_json_path = build_payload(meeting, transcript_path, transcript_text, transcriber)
    update_metadata(meeting, payload, transcript_path, document_json_path)
    ingest_meeting(meeting)

    print("\n[*] Generating Meeting Summary...")
    try:
        from app.services.summary import generate_meeting_summary
        summary_data = generate_meeting_summary(transcript_text)
        summary_target = meeting.meeting_dir / "summary.json"
        summary_target.write_text(json.dumps(summary_data, indent=4), encoding="utf-8")
        print(f"[*] Summary saved -> {summary_target}")
    except Exception as e:
        print(f"[!] Failed to generate summary: {e}")

    print("\nMeeting recovery is complete.")
    print(f"Meeting folder: {meeting.meeting_dir}")
    print(f"Transcript: {transcript_path}")
    print(f"Document JSON: {document_json_path}")
    print(f"Vector store: {meeting.vector_store_dir}")
    print("\nYou can also use `streamlit run app2.py` and open this meeting there.")

    if not args.skip_chat and prompt_yes_no("Open terminal chat for this meeting now?", default=True):
        chat_with_meeting(meeting)


if __name__ == "__main__":
    main()
