from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
try:
    from sarvamai import SarvamAI
except ImportError:
    SarvamAI = None


DEFAULT_MODEL = "saaras:v3"
DEFAULT_MODE = "codemix"
DEFAULT_LANGUAGE_CODE = "hi-IN"
DEFAULT_POLL_SECONDS = 5.0
SARVAM_API_KEY = "sk_5qbk6qgz_UkwlJ8W6W4lJeJ5nsCiyOaJy"


def make_output_name(input_path: str) -> str:
    dirname = os.path.dirname(input_path)
    base = os.path.splitext(os.path.basename(input_path))[0]
    filename = f"{base}_transcript.txt"
    return os.path.join(dirname, filename) if dirname else filename


def _resolve_api_key(explicit_api_key: str | None = None) -> str:
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()
    if SARVAM_API_KEY.strip():
        return SARVAM_API_KEY.strip()
    return os.environ.get("SARVAM_API_KEY", "").strip()


def _wait_for_job(job, poll_seconds: float):
    if hasattr(job, "wait_until_complete"):
        try:
            return job.wait_until_complete(poll_interval_seconds=poll_seconds)
        except TypeError:
            return job.wait_until_complete()

    if hasattr(job, "get_status"):
        while True:
            status = job.get_status()
            state = getattr(status, "job_state", None) or getattr(job, "job_state", None)
            if str(state).lower() in {"completed", "failed"}:
                return status
            time.sleep(poll_seconds)

    raise RuntimeError("The installed sarvamai SDK does not expose a supported wait method.")


def _job_failed(job, status) -> bool:
    if hasattr(job, "is_failed"):
        return bool(job.is_failed())

    state = getattr(status, "job_state", None) or getattr(job, "job_state", None)
    return str(state).lower() == "failed"


def _discover_downloaded_json(output_dir: Path) -> Path:
    json_files = sorted(output_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not json_files:
        raise FileNotFoundError(f"No JSON output found in {output_dir}")
    return json_files[0]


def _format_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "00:00:00"
    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _render_transcript_text(payload: dict) -> str:
    diarized = payload.get("diarized_transcript") or {}
    entries = diarized.get("entries") or []

    if entries:
        lines = []
        for entry in entries:
            transcript = str(entry.get("transcript", "")).strip()
            if not transcript:
                continue
            speaker_id = entry.get("speaker_id", "unknown")
            start_time = _format_seconds(entry.get("start_time_seconds"))
            end_time = _format_seconds(entry.get("end_time_seconds"))
            lines.append(f"[{start_time} - {end_time}] SPEAKER_{speaker_id}: {transcript}")
        if lines:
            return "\n".join(lines) + "\n"

    transcript = str(payload.get("transcript", "")).strip()
    if transcript:
        return transcript + "\n"

    timestamps = payload.get("timestamps") or {}
    words = timestamps.get("words") or []
    starts = timestamps.get("start_time_seconds") or []
    ends = timestamps.get("end_time_seconds") or []
    if words:
        lines = []
        for idx, word_text in enumerate(words):
            start_time = _format_seconds(starts[idx] if idx < len(starts) else None)
            end_time = _format_seconds(ends[idx] if idx < len(ends) else None)
            lines.append(f"[{start_time} - {end_time}] {word_text}")
        return "\n".join(lines) + "\n"

    return ""


def process_single_file(
    media_path: str,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    mode: str = DEFAULT_MODE,
    language_code: str = DEFAULT_LANGUAGE_CODE,
    num_speakers: int | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> str:
    if SarvamAI is None:
        raise RuntimeError("Missing dependency: sarvamai. Install it with `pip install sarvamai`.")

    resolved_api_key = _resolve_api_key(api_key)
    if not resolved_api_key:
        raise RuntimeError("Missing Sarvam API key. Set SARVAM_API_KEY or pass one explicitly.")

    media_file = Path(media_path)
    output_txt = Path(make_output_name(str(media_file)))

    client = SarvamAI(api_subscription_key=resolved_api_key)
    job_kwargs = {
        "model": model,
        "mode": mode,
        "language_code": language_code,
        "with_diarization": True,
    }
    if num_speakers:
        job_kwargs["num_speakers"] = num_speakers

    job = client.speech_to_text_job.create_job(**job_kwargs)
    job.upload_files(file_paths=[str(media_file)])
    job.start()

    status = _wait_for_job(job, poll_seconds)
    if _job_failed(job, status):
        raise RuntimeError(f"Sarvam job failed for {media_file.name}")

    if hasattr(job, "get_file_results"):
        file_results = job.get_file_results() or {}
        failed = file_results.get("failed") or []
        if failed:
            first_failure = failed[0]
            raise RuntimeError(
                f"{media_file.name} failed: {first_failure.get('error_message', 'unknown error')}"
            )

    with tempfile.TemporaryDirectory(prefix="sarvam_stt_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        job.download_outputs(output_dir=str(temp_dir))

        json_path = _discover_downloaded_json(temp_dir)
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

    transcript_text = _render_transcript_text(payload)
    if not transcript_text.strip():
        raise RuntimeError(f"No transcript text returned for {media_file.name}")

    output_txt.write_text(transcript_text, encoding="utf-8")
    return str(output_txt)
