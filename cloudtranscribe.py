"""
Transcription + Diarization Pipeline  (FULLY CLOUD)
────────────────────────────────────────────────────
Transcription : Groq API  (whisper-large-v3)
Diarization   : pyannote.ai Cloud API  (precision-2)

"""

import os
import sys
import math
import time
import uuid
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import requests
from pydub import AudioSegment
from groq import Groq

from dotenv import load_dotenv

load_dotenv()
# ─────────────────────────── CONFIG ───────────────────────────

SUPPORTED_EXTENSIONS = {
    ".mp4", ".mp3", ".m4a", ".wav", ".mkv", ".avi",
    ".flac", ".ogg", ".webm", ".mov", ".aac", ".wma",
}

def _load_groq_api_keys() -> list[str]:
    raw_keys = os.getenv("GROQ_Transcribe_API_KEY", "")
    if not raw_keys.strip():
        return []
    normalized = raw_keys.replace(";", ",")
    return [key.strip() for key in normalized.split(",") if key.strip()]


# This might work
GROQ_API_KEYS = _load_groq_api_keys()

#Else just paste the one i send here
# GROQ_API_KEYS = []


GROQ_MODEL = "whisper-large-v3"   # best available on Groq
CHUNK_MINUTES = 5                        # Groq file-size limit → chunk audio

# Pyannote.ai Cloud API
PYANNOTE_API_KEY = os.getenv("PYANNOTE_API_KEY")
PYANNOTE_API_BASE = "https://api.pyannote.ai/v1"
PYANNOTE_MODEL = "precision-2"           # best model (or "community-1" for free tier)
PYANNOTE_POLL_INTERVAL = 5               # seconds between polling for job status


# ─────────────────────────── PRETTY LOGGING ───────────────────

class Log:
    """Coloured, timestamped console logger."""

    _start = time.time()

    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    MAGENTA = "\033[95m"
    BG_GREEN = "\033[42m"
    BG_RED   = "\033[41m"

    @classmethod
    def _elapsed(cls) -> str:
        secs = time.time() - cls._start
        m, s = divmod(int(secs), 60)
        return f"{m:02d}:{s:02d}"

    @classmethod
    def _bar(cls, pct: int, width: int = 30) -> str:
        filled = int(width * pct / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"{cls.DIM}[{bar}]{cls.RESET}"

    @classmethod
    def step(cls, pct: int, msg: str):
        print(f"  {cls._bar(pct)} {cls.CYAN}{cls._elapsed()}{cls.RESET}  {cls.BOLD}{msg}{cls.RESET}")

    @classmethod
    def info(cls, msg: str):
        print(f"  {cls.DIM}{'>':>39}{cls.RESET} {cls.DIM}↳ {msg}{cls.RESET}")

    @classmethod
    def success(cls, msg: str):
        print(f"  {cls.GREEN}{'':>39}✓ {msg}{cls.RESET}")

    @classmethod
    def warn(cls, msg: str):
        print(f"  {cls.YELLOW}{'':>39}⚠ {msg}{cls.RESET}")

    @classmethod
    def error(cls, msg: str):
        print(f"  {cls.RED}{'':>39}✗ {msg}{cls.RESET}")

    @classmethod
    def header(cls):
        print()
        print(f"  {cls.MAGENTA}{cls.BOLD}╔══════════════════════════════════════════════════╗{cls.RESET}")
        print(f"  {cls.MAGENTA}{cls.BOLD}║   TRANSCRIPTION + DIARIZATION PIPELINE           ║{cls.RESET}")
        print(f"  {cls.MAGENTA}{cls.BOLD}║   Groq Whisper  ·  Pyannote.ai Cloud              ║{cls.RESET}")
        print(f"  {cls.MAGENTA}{cls.BOLD}╚══════════════════════════════════════════════════╝{cls.RESET}")
        print()

    @classmethod
    def file_banner(cls, idx: int, total: int, filename: str, size_mb: float):
        print()
        print(f"  {cls.CYAN}{cls.BOLD}┌──────────────────────────────────────────────────┐{cls.RESET}")
        print(f"  {cls.CYAN}{cls.BOLD}│  FILE {idx}/{total}: {filename:<39}{cls.RESET}{cls.CYAN}{cls.BOLD}│{cls.RESET}")
        print(f"  {cls.CYAN}{cls.BOLD}│  Size: {size_mb:.1f} MB{' ' * max(0, 38 - len(f'{size_mb:.1f} MB'))}{cls.RESET}{cls.CYAN}{cls.BOLD}│{cls.RESET}")
        print(f"  {cls.CYAN}{cls.BOLD}└──────────────────────────────────────────────────┘{cls.RESET}")
        cls._start = time.time()  # reset timer per file

    @classmethod
    def file_footer(cls, output_path: str):
        print()
        print(f"  {cls.BG_GREEN}{cls.BOLD} DONE {cls.RESET}  Transcript saved → {cls.BOLD}{output_path}{cls.RESET}")
        print(f"  {cls.DIM}File time: {cls._elapsed()}{cls.RESET}")

    @classmethod
    def footer(cls, total_files: int, total_time: float):
        m, s = divmod(int(total_time), 60)
        print()
        print(f"  {cls.BG_GREEN}{cls.BOLD} ALL DONE {cls.RESET}  Processed {cls.BOLD}{total_files}{cls.RESET} file(s) in {cls.BOLD}{m:02d}:{s:02d}{cls.RESET}")
        print()


# ─────────────────────────── API KEY MANAGER ──────────────────

class GroqKeyManager:
    """Round-robin Groq API key rotation on rate-limit errors."""

    def __init__(self, keys: list):
        if not keys:
            raise ValueError("At least one Groq API key must be provided in GROQ_API_KEYS!")
        self.keys = keys
        self._idx = 0

    @property
    def current_key(self) -> str:
        return self.keys[self._idx]

    def rotate(self) -> str:
        """Move to the next key and return it."""
        self._idx = (self._idx + 1) % len(self.keys)
        return self.current_key

    @property
    def total_keys(self) -> int:
        return len(self.keys)


# Global key manager instance
_key_manager = GroqKeyManager(GROQ_API_KEYS)


# ─────────────────────────── DATA STRUCTURES ──────────────────

@dataclass
class ASRSegment:
    start: float
    end: float
    text: str

@dataclass
class SpkSegment:
    start: float
    end: float
    speaker: str


# ─────────────────────────── UTILITIES ────────────────────────

def ensure_ffmpeg():
    from shutil import which
    if which("ffmpeg") is None:
        Log.error("ffmpeg not found on PATH — install it first!")
        sys.exit(1)


def scan_for_media(directory: str = ".") -> List[str]:
    """Find all supported audio/video files in the given directory."""
    files = []
    for entry in sorted(os.listdir(directory)):
        ext = os.path.splitext(entry)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            full_path = os.path.join(directory, entry)
            if os.path.isfile(full_path):
                files.append(full_path)
    return files


def make_output_name(input_path: str) -> str:
    """Generate output transcript filename from input: video.mp4 → video_transcript.txt"""
    base = os.path.splitext(os.path.basename(input_path))[0]
    return f"{base}_transcript.txt"


def extract_audio_to_wav(input_path: str, output_wav: str):
    """Extract mono 16 kHz WAV from any media file via ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        output_wav,
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to extract audio from {input_path}")


def fmt_duration(seconds: float) -> str:
    """Human-friendly duration string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


# ─────────────────────────── GROQ TRANSCRIPTION ───────────────

def transcribe_with_groq(audio_path: str) -> List[ASRSegment]:
    """
    Transcribe audio via Groq API using whisper-large-v3.
    Handles chunking for files over the Groq size limit.
    Returns segment-level segments with timestamps.
    """
    Log.step(10, "Preparing audio for Groq transcription …")

    ensure_ffmpeg()

    # Convert to WAV first for consistent handling
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    extract_audio_to_wav(audio_path, tmp_wav)

    audio = AudioSegment.from_wav(tmp_wav)
    duration_sec = len(audio) / 1000.0
    Log.info(f"Audio duration: {fmt_duration(duration_sec)}")

    chunk_ms = CHUNK_MINUTES * 60 * 1000
    total_chunks = math.ceil(len(audio) / chunk_ms)
    Log.info(f"Splitting into {total_chunks} chunk(s) of {CHUNK_MINUTES} min each")

    all_segments: List[ASRSegment] = []

    for i in range(total_chunks):
        pct = 10 + int(40 * (i / total_chunks))  # progress 10→50%

        start_ms = i * chunk_ms
        end_ms = min((i + 1) * chunk_ms, len(audio))
        chunk = audio[start_ms:end_ms]

        chunk_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        chunk.export(chunk_path, format="mp3")

        Log.step(pct, f"Groq → transcribing chunk {i+1}/{total_chunks}")

        # Retry loop: rotate API keys on rate-limit errors
        attempts = 0
        max_attempts = _key_manager.total_keys  # try each key at most once
        success = False

        while attempts < max_attempts:
            client = Groq(api_key=_key_manager.current_key)
            t0 = time.time()
            try:
                with open(chunk_path, "rb") as f:
                    result = client.audio.transcriptions.create(
                        model=GROQ_MODEL,
                        file=f,
                        language="en",
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                    )

                elapsed = time.time() - t0
                Log.success(f"Chunk {i+1} done in {elapsed:.1f}s (key #{_key_manager._idx + 1})")

                # Parse segments with timestamp offset
                offset_sec = start_ms / 1000.0
                if hasattr(result, "segments") and result.segments:
                    for seg in result.segments:
                        all_segments.append(ASRSegment(
                            start=seg.get("start", seg["start"]) + offset_sec,
                            end=seg.get("end", seg["end"]) + offset_sec,
                            text=seg.get("text", "").strip(),
                        ))
                else:
                    # Fallback: treat entire chunk as one segment
                    all_segments.append(ASRSegment(
                        start=offset_sec,
                        end=offset_sec + (end_ms - start_ms) / 1000.0,
                        text=result.text.strip() if hasattr(result, "text") else str(result).strip(),
                    ))
                success = True
                break  # chunk done, move on

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = ("rate_limit" in err_str
                                 or "429" in err_str
                                 or "rate limit" in err_str
                                 or "too many requests" in err_str)
                is_invalid_key = ("401" in err_str
                                  or "authentication" in err_str
                                  or "invalid_api_key" in err_str
                                  or "invalid api key" in err_str
                                  or "unauthorized" in err_str
                                  or "invalid x-api-key" in err_str)
                is_retryable = is_rate_limit or is_invalid_key

                if is_retryable and attempts + 1 < max_attempts:
                    old_key_num = _key_manager._idx + 1
                    reason = "Rate limit" if is_rate_limit else "Invalid API key"
                    new_key = _key_manager.rotate()
                    new_key_num = _key_manager._idx + 1
                    Log.warn(f"{reason} on key #{old_key_num} — rotating to key #{new_key_num}")
                    attempts += 1
                    time.sleep(1)  # brief pause before retrying
                else:
                    Log.error(f"Chunk {i+1} failed: {e}")
                    os.remove(chunk_path)
                    raise

        if not success:
            os.remove(chunk_path)
            raise RuntimeError(f"All {max_attempts} API keys exhausted for chunk {i+1}")

        os.remove(chunk_path)

    # Cleanup
    os.remove(tmp_wav)

    Log.step(50, f"Transcription complete — {len(all_segments)} segments")
    return all_segments


# ─────────────────────────── PYANNOTE.AI CLOUD DIARIZATION ────

def _pyannote_headers() -> dict:
    """Build authorization headers for pyannote.ai API."""
    return {
        "Authorization": f"Bearer {PYANNOTE_API_KEY}",
        "Content-Type": "application/json",
    }


def _upload_to_pyannote(wav_path: str) -> str:
    """
    Upload a local WAV file to pyannote.ai temporary storage.
    Returns the media:// URL to reference in diarization jobs.
    """
    # Generate a unique media key
    media_key = f"media://upload-{uuid.uuid4().hex[:12]}"

    Log.info(f"Requesting upload URL for {media_key} …")

    # Step 1: Declare the media:// URL and get a pre-signed upload URL
    resp = requests.post(
        f"{PYANNOTE_API_BASE}/media/input",
        headers=_pyannote_headers(),
        json={"url": media_key},
    )
    resp.raise_for_status()
    upload_url = resp.json()["url"]

    # Step 2: PUT the actual file data to the pre-signed URL
    Log.info("Uploading audio to pyannote.ai temporary storage …")
    file_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
    Log.info(f"Upload size: {file_size_mb:.1f} MB")

    with open(wav_path, "rb") as f:
        put_resp = requests.put(
            upload_url,
            data=f,
            headers={"Content-Type": "audio/wav"},
        )
    put_resp.raise_for_status()

    Log.success("Audio uploaded to pyannote.ai storage")
    return media_key


def _submit_diarization_job(media_url: str) -> str:
    """
    Submit a diarization job to pyannote.ai.
    Returns the jobId for polling.
    """
    payload = {
        "url": media_url,
        "model": PYANNOTE_MODEL,
    }

    resp = requests.post(
        f"{PYANNOTE_API_BASE}/diarize",
        headers=_pyannote_headers(),
        json=payload,
    )
    resp.raise_for_status()
    data = resp.json()
    job_id = data["jobId"]
    Log.success(f"Diarization job submitted — ID: {job_id}")
    return job_id


def _poll_job_until_done(job_id: str) -> dict:
    """
    Poll pyannote.ai for job status until it completes.
    Returns the full job result dict on success.
    """
    headers = {
        "Authorization": f"Bearer {PYANNOTE_API_KEY}",
    }

    poll_count = 0
    while True:
        resp = requests.get(
            f"{PYANNOTE_API_BASE}/jobs/{job_id}",
            headers=headers,
        )
        resp.raise_for_status()
        result = resp.json()
        status = result.get("status", "unknown")

        poll_count += 1
        if status == "succeeded":
            Log.success(f"Diarization job completed (polled {poll_count} times)")
            return result
        elif status in ("failed", "canceled"):
            error_msg = result.get("output", {}).get("error", "Unknown error")
            raise RuntimeError(f"Diarization job {status}: {error_msg}")
        else:
            # Still running — show a spinner-like update
            if poll_count % 3 == 1:
                Log.info(f"Job status: {status}  (poll #{poll_count}, waiting {PYANNOTE_POLL_INTERVAL}s …)")
            time.sleep(PYANNOTE_POLL_INTERVAL)


def diarize_audio_cloud(audio_path: str) -> List[SpkSegment]:
    """
    Run speaker diarization via pyannote.ai cloud API.
    Uploads the audio, submits a job, polls for results.
    Returns labelled speaker segments.
    """
    Log.step(55, "Starting cloud diarization via pyannote.ai …")

    ensure_ffmpeg()

    # Convert to WAV for pyannote upload
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        extract_audio_to_wav(audio_path, tmp_wav)

        # Upload to pyannote temp storage
        Log.step(58, "Uploading audio to pyannote.ai …")
        media_url = _upload_to_pyannote(tmp_wav)

        # Submit diarization job
        Log.step(62, "Submitting diarization job …")
        job_id = _submit_diarization_job(media_url)

        # Poll until completion
        Log.step(65, "Waiting for diarization results …")
        result = _poll_job_until_done(job_id)

        # Parse output segments
        output = result.get("output", {})
        diarization_data = output.get("diarization", [])

        if not diarization_data:
            Log.warn("No diarization segments returned — audio may be too short or silent")
            return []

        spk_segments: List[SpkSegment] = []
        for seg in diarization_data:
            spk_segments.append(SpkSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                speaker=seg["speaker"],
            ))

        # Identify unique speakers (pyannote.ai already uses SPEAKER_XX format)
        unique_speakers = sorted(set(s.speaker for s in spk_segments))

        Log.step(80, f"Diarization complete — {len(spk_segments)} turns, {len(unique_speakers)} speakers")
        for spk in unique_speakers:
            count = sum(1 for s in spk_segments if s.speaker == spk)
            Log.info(f"  {spk}: {count} turns")

        return spk_segments

    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


# ─────────────────────────── MERGE & OUTPUT ───────────────────

def find_speaker_for_time(spk_segments: List[SpkSegment], t: float) -> Optional[str]:
    """Binary-search-style lookup (segments are sorted by start time)."""
    lo, hi = 0, len(spk_segments) - 1
    while lo <= hi:
        mid_idx = (lo + hi) // 2
        seg = spk_segments[mid_idx]
        if t < seg.start:
            hi = mid_idx - 1
        elif t > seg.end:
            lo = mid_idx + 1
        else:
            return seg.speaker
    return None


def assign_speakers(asr_segments: List[ASRSegment], spk_segments: List[SpkSegment]) -> List[str]:
    """Match each ASR segment to the best-overlapping speaker."""
    Log.step(85, "Assigning speakers to transcript segments …")

    # Sort diarization segments for binary search
    spk_segments_sorted = sorted(spk_segments, key=lambda s: s.start)

    lines: List[str] = []
    last_speaker = "UNKNOWN"
    speaker_line_counts: dict = {}

    for seg in asr_segments:
        mid = (seg.start + seg.end) / 2.0
        speaker = find_speaker_for_time(spk_segments_sorted, mid)

        if speaker is None:
            speaker = last_speaker
        else:
            last_speaker = speaker

        speaker_line_counts[speaker] = speaker_line_counts.get(speaker, 0) + 1
        lines.append(f"{speaker}: {seg.text}")

    Log.step(90, "Speaker assignment complete")
    for spk, count in sorted(speaker_line_counts.items()):
        Log.info(f"  {spk}: {count} lines")

    return lines


def write_transcript(lines: List[str], output_path: str):
    Log.step(95, f"Writing transcript → {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    size_kb = os.path.getsize(output_path) / 1024
    Log.success(f"Saved {len(lines)} lines ({size_kb:.1f} KB)")


# ─────────────────────────── MAIN ─────────────────────────────

def process_single_file(audio_path: str):
    """Full pipeline for one audio/video file."""
    output_path = make_output_name(audio_path)

    # Skip if transcript already exists
    if os.path.exists(output_path):
        Log.warn(f"Transcript already exists: {output_path} — skipping (delete it to re-process)")
        return

    # Transcribe via Groq API (cloud — fast)
    asr_segments = transcribe_with_groq(audio_path)

    # Diarize via pyannote.ai Cloud API (no local GPU needed!)
    spk_segments = diarize_audio_cloud(audio_path)

    # Merge + write
    lines = assign_speakers(asr_segments, spk_segments)
    write_transcript(lines, output_path)

    Log.file_footer(output_path)


def main():
    Log.header()
    global_start = time.time()

    # 1. Scan for media files
    Log.step(2, "Scanning current folder for media files …")
    media_files = scan_for_media(".")

    if not media_files:
        Log.error("No supported media files found in current directory!")
        Log.info(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    Log.success(f"Found {len(media_files)} file(s):")
    for f in media_files:
        size = os.path.getsize(f) / (1024 * 1024)
        Log.info(f"  {os.path.basename(f)} ({size:.1f} MB)")

    # 2. Process each file (no need to pre-load anything — all cloud!)
    processed = 0
    for idx, audio_file in enumerate(media_files, 1):
        size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        Log.file_banner(idx, len(media_files), os.path.basename(audio_file), size_mb)

        try:
            process_single_file(audio_file)
            processed += 1
        except Exception as e:
            Log.error(f"Failed to process {os.path.basename(audio_file)}: {e}")
            Log.info("Continuing with next file …")
            continue

    Log.footer(processed, time.time() - global_start)


if __name__ == "__main__":
    main()
