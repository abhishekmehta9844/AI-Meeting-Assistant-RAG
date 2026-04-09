import json
import os
import re
import sys
import threading
import time
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
try:
    import pyaudiowpatch as pyaudio
except ImportError:
    pyaudio = None
import soundfile as sf
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

import cloudtranscribe
import sarvam_transcribe
from ingest import ingest
from meeting_library import get_meeting_dir, write_meeting_metadata

PREFERRED_LOOPBACK_DEVICE: str | None = None


def find_best_loopback_wasapi(pa):
    if pyaudio is None:
        raise RuntimeError(
            "pyaudiowpatch is not installed. Run `pip install pyaudiowpatch` and try again."
        )

    try:
        wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    except Exception as exc:
        raise RuntimeError(f"WASAPI host API is not available: {exc}") from exc

    default_out_idx = wasapi_info["defaultOutputDevice"]
    default_out = pa.get_device_info_by_index(default_out_idx)
    print(f"[*] Default output device: {default_out['name']}")

    loopback_devices = []
    for index in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(index)
        if info.get("isLoopbackDevice"):
            loopback_devices.append((index, info))

    if not loopback_devices:
        raise RuntimeError(
            "No WASAPI loopback device found. Enable 'Stereo Mix' or install VB-Cable."
        )

    print(f"[*] Found {len(loopback_devices)} WASAPI loopback device(s):")
    for _, info in loopback_devices:
        print(f"      - {info['name']}")

    if PREFERRED_LOOPBACK_DEVICE:
        preferred = next(
            (
                (index, info)
                for index, info in loopback_devices
                if PREFERRED_LOOPBACK_DEVICE.lower() in info["name"].lower()
            ),
            None,
        )
        if preferred is not None:
            print(f"[*] Using preferred loopback device: {preferred[1]['name']}")
            return preferred
        print(f"[!] Preferred device '{PREFERRED_LOOPBACK_DEVICE}' not found. Falling back to default output loopback.")

    default_name = default_out["name"].lower()
    exact = next(
        (
            (index, info)
            for index, info in loopback_devices
            if default_name in info["name"].lower() or info["name"].lower() in default_name
        ),
        None,
    )
    if exact is not None:
        print(f"[*] Found loopback device for default output: {exact[1]['name']}")
        return exact

    print(f"[*] Using fallback loopback device: {loopback_devices[0][1]['name']}")
    return loopback_devices[0]


# ────────────────────────────────────────────────────────────────────────────

class MeetingBot:
    def __init__(self, meeting_url: str, transcription_engine: str = "whisper"):
        self.meeting_url = meeting_url
        self.transcription_engine = transcription_engine
        self.started_at = datetime.now()
        meeting_slug = re.sub(r"[^a-zA-Z0-9]+", "-", meeting_url).strip("-").lower()[-40:] or "google-meet"
        self.meeting_id = f"{self.started_at.strftime('%Y%m%d_%H%M%S')}_{meeting_slug}"
        self.meeting_dir = get_meeting_dir(self.meeting_id)
        self.meeting_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir = self.meeting_dir / "documents"
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir = self.meeting_dir / "vector_store"
        self.collection_name = f"meeting_{self.started_at.strftime('%Y%m%d_%H%M%S')}"
        self.recording = False
        self.recording_started = threading.Event()
        self.recording_error = None
        self.stop_recording = threading.Event()
        self.audio_file = str(self.meeting_dir / f"{self.meeting_id}.wav")
        self._peak_lock = threading.Lock()
        self._last_peak: float = 0.0

    @staticmethod
    def _extract_participants(transcript_text: str) -> list[str]:

        participants = []
        seen = set()

        for raw_line in transcript_text.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue

            speaker = line.split(":", 1)[0].strip()
            if not speaker:
                continue

            normalized = speaker.lower()
            if normalized in seen:
                continue

            seen.add(normalized)
            participants.append(speaker)

        return participants

    # ── Audio recording ───────────────────────────────────────────────────
    def _record_audio(self):
        print(f"[*] Starting audio recording (WASAPI Loopback) -> {self.audio_file}")
        self.recording = True
        try:
            if pyaudio is None:
                raise RuntimeError(
                    "pyaudiowpatch is not installed. Run `pip install pyaudiowpatch` and try again."
                )

            pa = pyaudio.PyAudio()
            stream = None
            audio_chunks: list[np.ndarray] = []

            try:
                device_idx, device_info = find_best_loopback_wasapi(pa)
                samplerate = int(device_info["defaultSampleRate"])
                channels = min(int(device_info["maxInputChannels"]), 2)
                if channels < 1:
                    raise RuntimeError(f"Selected loopback device has no input channels: {device_info['name']}")

                print(f"[*] Recording: {device_info['name']} | {samplerate} Hz | {channels} ch")

                frames_per_buffer = max(1024, int(samplerate * 0.1))

                def callback(in_data, frame_count, time_info, status):
                    if in_data:
                        chunk = np.frombuffer(in_data, dtype=np.float32).copy()
                        if channels > 1:
                            chunk = chunk.reshape(-1, channels)
                            energies = [float(np.sum(chunk[:, channel] ** 2)) for channel in range(channels)]
                            chunk = chunk[:, int(np.argmax(energies))]

                        peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
                        with self._peak_lock:
                            self._last_peak = peak
                        audio_chunks.append(chunk)
                    return (None, pyaudio.paContinue)

                stream = pa.open(
                    format=pyaudio.paFloat32,
                    channels=channels,
                    rate=samplerate,
                    input=True,
                    input_device_index=device_idx,
                    frames_per_buffer=frames_per_buffer,
                    stream_callback=callback
                )

                self.recording_started.set()
                stream.start_stream()

                while not self.stop_recording.is_set():
                    time.sleep(0.2)

                if audio_chunks:
                    all_audio = np.concatenate(audio_chunks).astype(np.float32, copy=False)
                    peak = float(np.max(np.abs(all_audio))) if all_audio.size else 0.0
                    if peak > 0.99:
                        all_audio = all_audio / peak
                    sf.write(self.audio_file, all_audio, samplerate, subtype="PCM_16")
                    print(f"\n[*] Saved {len(all_audio) / samplerate:.1f}s of audio -> {self.audio_file}")
                else:
                    print("\n[!] No audio data captured.")
            finally:
                if stream is not None:
                    try:
                        stream.stop_stream()
                    except Exception:
                        pass
                    try:
                        stream.close()
                    except Exception:
                        pass
                pa.terminate()
        except Exception as exc:
            self.recording_error = exc
            self.recording_started.set()   
            print(f"\n[!] Audio recording failed: {exc}")
        finally:
            self.recording = False
            print("\n[*] Audio recording stopped.")

    def _peak_monitor(self):
        """
        Prints a live audio-level bar in the terminal every second.
        A bar with non-zero blocks means audio IS being captured.
        If the bar stays empty the whole time, audio is NOT reaching the recorder.
        """
        while not self.stop_recording.is_set():
            time.sleep(1)
            with self._peak_lock:
                peak = self._last_peak
            bar_len = min(30, int(peak * 150))
            bar = "█" * bar_len + "░" * (30 - bar_len)
            status = "SILENT" if bar_len == 0 else "AUDIO "
            print(f"\r  [{bar}] {status}  peak={peak:.4f}", end="", flush=True)

    # ── Google Meet automation ─────────────────────────────────────────────

    def _dismiss_prejoin_popups(self, page) -> None:
        popup_selectors = [
            'button:has-text("Got it")',
            'button:has-text("OK")',
            'button:has-text("Continue without microphone and camera")',
            'button[aria-label="Close"]',
            '[role="button"]:has-text("Got it")',
        ]
        for _ in range(4):
            clicked_any = False
            for selector in popup_selectors:
                try:
                    locator = page.locator(selector).first
                    locator.wait_for(state="visible", timeout=1200)
                    locator.click(timeout=2000)
                    page.wait_for_timeout(600)
                    clicked_any = True
                    print(f"[*] Dismissed popup: {selector}")
                except Exception:
                    continue
            if not clicked_any:
                break

    def _turn_prejoin_control_off(self, page, control_name: str) -> bool:
        label_patterns = [
            re.compile(rf"turn off {control_name}", re.IGNORECASE),
            re.compile(rf"turn on {control_name}", re.IGNORECASE),
            re.compile(control_name, re.IGNORECASE),
        ]

        for pattern in label_patterns:
            try:
                locator = page.get_by_role("button", name=pattern).first
                locator.wait_for(state="visible", timeout=2500)
                aria_label = locator.get_attribute("aria-label") or ""
                title = locator.get_attribute("title") or ""
                label_text = f"{aria_label} {title}".strip().lower()

                if f"turn on {control_name}" in label_text:
                    print(f"[*] {control_name.capitalize()} already off.")
                    return True

                locator.click(timeout=2500)
                page.wait_for_timeout(500)
                print(f"[*] Turned off {control_name}.")
                return True
            except Exception:
                continue

        print(f"[!] Could not find the {control_name} toggle on the prejoin screen.")
        return False

    def _disable_microphone_and_camera(self, page) -> None:
        print("[*] Muting bot camera and microphone...")

        try:
            page.locator("body").click(timeout=2000)
            page.wait_for_timeout(200)
            page.keyboard.press("Control+d")
            page.wait_for_timeout(400)
            page.keyboard.press("Control+e")
            page.wait_for_timeout(400)
        except Exception as exc:
            print(f"[!] Could not mute via shortcut: {exc}")

        self._turn_prejoin_control_off(page, "microphone")
        self._turn_prejoin_control_off(page, "camera")

    def _enter_guest_name(self, page) -> None:
        for selector in ['input[aria-label="Your name"]', 'input[placeholder="Your name"]', 'input[type="text"]']:
            try:
                inp = page.locator(selector).first
                inp.wait_for(state="visible", timeout=1500)
                inp.fill("AI Meeting Bot", timeout=3000)
                page.wait_for_timeout(500)
                print("[*] Entered bot name.")
                return
            except Exception:
                continue

    def _click_join_button(self, page) -> bool:
        for selector in [
            'button:has-text("Ask to join")',
            'button:has-text("Join now")',
            '[role="button"]:has-text("Ask to join")',
            '[role="button"]:has-text("Join now")',
        ]:
            try:
                btn = page.locator(selector).first
                btn.wait_for(state="visible", timeout=5000)
                btn.click(timeout=4000)
                print("[*] Clicked join button.")
                return True
            except Exception:
                continue
        return False

    def _wait_until_joined(self, page) -> None:
        for selector in [
            'button[aria-label="Leave call"]',
            'button[aria-label*="Leave call"]',
            '[role="button"][aria-label*="Leave call"]',
        ]:
            try:
                page.wait_for_selector(selector, state="visible", timeout=120000)
                return
            except PlaywrightTimeoutError:
                continue

        body = page.locator("body").inner_text(timeout=5000)
        if "can't join this video call" in body.lower():
            raise RuntimeError("Google Meet rejected the join request.")
        raise RuntimeError("Timed out waiting to be admitted to the meeting.")

    def _leave_meeting(self, page) -> None:
        print("[*] Leaving meeting...")
        for selector in [
            'button[aria-label="Leave call"]',
            'button[aria-label*="Leave call"]',
            '[role="button"][aria-label*="Leave call"]',
        ]:
            try:
                btn = page.locator(selector).first
                btn.wait_for(state="visible", timeout=1500)
                btn.click(timeout=2500)
                page.wait_for_timeout(1500)
                return
            except Exception:
                continue

    # ── Main flow ──────────────────────────────────────────────────────────

    def run(self):
        print(f"[*] Launching bot for: {self.meeting_url}")
        audio_thread = monitor_thread = None

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=False,
                args=[
                    "--use-fake-ui-for-media-stream",
                    "--disable-blink-features=AutomationControlled",
                ],
            )

            context = browser.new_context(permissions=["camera", "microphone"])
            page = context.new_page()

            try:
                print("[*] Navigating to Google Meet...")
                page.goto(self.meeting_url, wait_until="domcontentloaded", timeout=120000)
                page.wait_for_timeout(5000)

                self._dismiss_prejoin_popups(page)
                self._disable_microphone_and_camera(page)
                self._enter_guest_name(page)
                self._dismiss_prejoin_popups(page)

                joined = self._click_join_button(page)
                if not joined:
                    print("[!] Join button not found — please click it manually in the browser.")

                print("[*] Waiting to be admitted...")
                self._wait_until_joined(page)
                print("[*] Joined the meeting!")

                page.wait_for_timeout(3000)

                # ── Start recording ────────────────────────────────────────
                audio_thread = threading.Thread(target=self._record_audio, daemon=True)
                audio_thread.start()

                print("[*] Waiting for audio recorder to initialise...")
                if not self.recording_started.wait(timeout=15):
                    raise RuntimeError("Audio recorder did not start within 15 s.")

                if self.recording_error is not None:
                    raise RuntimeError(f"Recorder failed: {self.recording_error}")

                print("[*] Audio recorder is live.")
                print()
                print("  Live audio meter (non-empty bar = audio IS being captured):")

                monitor_thread = threading.Thread(target=self._peak_monitor, daemon=True)
                monitor_thread.start()

                print()
                print("=" * 55)
                print("  MEETING IS BEING RECORDED.")
                print("  Press Enter to STOP recording and begin transcription.")
                print("=" * 55)
                print()

                input()

            except Exception as exc:
                print(f"\n[!] Error: {exc}")
            finally:
                self.stop_recording.set()
                self.recording = False
                if audio_thread:
                    audio_thread.join(timeout=10)
                if monitor_thread:
                    monitor_thread.join(timeout=3)

                try:
                    self._leave_meeting(page)
                except Exception:
                    pass

                browser.close()
                self._process_meeting()

    # ── Post-processing ────────────────────────────────────────────────────

    def _process_meeting(self):
        if self.recording_error is not None:
            print("[!] Skipping transcription — recording failed.")
            return
        if not self.recording_started.is_set():
            print("[!] Recording never started — nothing to transcribe.")
            return
        if not os.path.exists(self.audio_file):
            print("[!] Audio file not found.")
            return
        if os.path.getsize(self.audio_file) < 1000:
            print("[!] Audio file is too small — nothing was recorded.")
            return

        print("\n[*] Initialising cloud transcription pipeline...")
        try:
            if self.transcription_engine == "sarvam":
                sarvam_transcribe.process_single_file(self.audio_file)
            else:
                cloudtranscribe.process_single_file(self.audio_file)
        except Exception as exc:
            print(f"[!] Transcription failed: {exc}")
            return

        if self.transcription_engine == "sarvam":
            transcript_file = sarvam_transcribe.make_output_name(self.audio_file)
        else:
            transcript_file = cloudtranscribe.make_output_name(self.audio_file)
        if not os.path.exists(transcript_file):
            print("[!] Transcript file was not generated.")
            return

        print(f"[*] Transcript saved -> {transcript_file}. Preparing ingest...")
        with open(transcript_file, "r", encoding="utf-8") as fh:
            content = fh.read()
        participants = self._extract_participants(content)

        json_target = self.documents_dir / f"{Path(transcript_file).stem}.json"

        payload = {
            "title": f"Meeting {self.started_at.strftime('%Y-%m-%d %H:%M')}",
            "content": content,
            "source_audio": str(Path(self.audio_file).name),
            "source_transcript": str(Path(transcript_file).name),
            "type": "meeting_transcript",
            "meeting_id": self.meeting_id,
            "meeting_url": self.meeting_url,
            "started_at": self.started_at.isoformat(),
            "participants": participants,
            "transcription_engine": self.transcription_engine,
        }
        with open(json_target, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=4)

        write_meeting_metadata(
            self.meeting_dir,
            {
                "meeting_id": self.meeting_id,
                "title": payload["title"],
                "meeting_url": self.meeting_url,
                "started_at": self.started_at.isoformat(),
                "audio_file": str(Path(self.audio_file).name),
                "transcript_file": str(Path(transcript_file).name),
                "document_json": str(json_target.relative_to(self.meeting_dir)),
                "vector_store_dir": str(self.vector_store_dir),
                "collection_name": self.collection_name,
                "transcript_preview": content[:300],
                "participants": participants,
                "transcription_engine": self.transcription_engine,
            },
        )

        print(f"[*] JSON saved -> {json_target}")
        print(f"[*] Meeting assets saved in -> {self.meeting_dir}")
        print("[*] Starting Vector Store Ingestion for this meeting only...")
        ingest(
            pdf_dir=None,
            json_dir=str(self.documents_dir),
            persist_dir=str(self.vector_store_dir),
            collection_name=self.collection_name,
            reset_collection=True,
        )
        print("\n[*] Bot sequence complete!")
        print("\n*** RUN `streamlit run app2.py` TO CHAT WITH YOUR MEETING ***\n")


# ── Entry point ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Join a Google Meet, record it, transcribe it, and ingest it.")
    parser.add_argument("meeting_url", help="Google Meet URL")
    parser.add_argument(
        "--transcriber",
        choices=["whisper", "sarvam"],
        default=None,
        help="Transcription engine to use for this meeting.",
    )
    return parser.parse_args()


def choose_transcriber(explicit_choice: str | None) -> str:
    if explicit_choice:
        return explicit_choice

    print("Choose transcription engine:")
    print("1. whisper")
    print("2. sarvam")
    while True:
        choice = input("Enter 1 or 2 [default 1]: ").strip().lower()
        if choice in {"", "1", "whisper"}:
            return "whisper"
        if choice in {"2", "sarvam"}:
            return "sarvam"
        print("Please enter 1 for whisper or 2 for sarvam.")


if __name__ == "__main__":
    args = parse_args()
    transcriber = choose_transcriber(args.transcriber)

    print()
    print(f"[*] Selected transcription engine: {transcriber}")
    print("Tip: if the live audio meter stays empty the whole time,")
    print("     set PREFERRED_LOOPBACK_DEVICE at the top of this file")
    print("     to the name of the device Chrome is actually using.")

    bot = MeetingBot(args.meeting_url, transcription_engine=transcriber)
    bot.run()
