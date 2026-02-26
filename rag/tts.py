"""
TTS support via edge-tts (Microsoft Neural voices).
Playback: Windows uses built-in winmm (no extra deps), macOS uses afplay,
Linux uses ffplay/mpv.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional


def speak(
    text: str,
    voice: Optional[str] = None,
    save_to: Optional[Path] = None,
    max_chars: int = 0,
) -> None:
    """
    Synthesize *text* and either play it or save it to an MP3 file.

    Args:
        text:      The text to synthesize.
        voice:     edge-tts voice name (e.g. ``it-IT-ElsaNeural``).
                   Falls back to ``RAG_TTS_VOICE`` env-var / config default.
        save_to:   If given, save the MP3 to this path instead of playing.
        max_chars: Truncate text to this many characters (0 = no limit).
    """
    from . import config

    voice = voice or config.TTS_VOICE
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]

    asyncio.run(_speak_async(text, voice, save_to))


async def _speak_async(
    text: str, voice: str, save_to: Optional[Path]
) -> None:
    try:
        import edge_tts  # type: ignore
    except ImportError:
        raise RuntimeError(
            "edge-tts is not installed. Run: pip install edge-tts"
        )

    communicate = edge_tts.Communicate(text, voice)

    if save_to:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        await communicate.save(str(save_to))
        return

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp = f.name
    try:
        await communicate.save(tmp)
        _play(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _play(path: str) -> None:
    """Play an MP3 file, blocking until playback finishes."""
    if sys.platform == "win32":
        import ctypes
        mci: ctypes.CDLL = ctypes.windll.winmm  # type: ignore[attr-defined]
        alias = "rag_tts"
        mci.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, None)
        mci.mciSendStringW(f"play {alias} wait", None, 0, None)
        mci.mciSendStringW(f"close {alias}", None, 0, None)
    elif sys.platform == "darwin":
        import subprocess
        subprocess.run(["afplay", path], check=True)
    else:
        import subprocess
        # Try ffplay first, fall back to mpv
        for player in (["ffplay", "-nodisp", "-autoexit", path],
                       ["mpv", "--no-video", path]):
            try:
                subprocess.run(
                    player,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                break
            except FileNotFoundError:
                continue
        else:
            raise RuntimeError(
                "No audio player found. Install ffplay (ffmpeg) or mpv."
            )


# ── text extraction helpers ────────────────────────────────────────────────────

def extract_text(path: Path) -> str:
    """Extract plain text from a .txt, .md, .pdf, or .json file."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix == ".json":
        return _extract_json(path)
    else:
        # txt / md / any other text file
        return path.read_text(encoding="utf-8", errors="replace")


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf is not installed. Run: pip install pypdf")

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _extract_json(path: Path) -> str:
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    # If it's the output format saved by _save_output (has "answer" key), read that.
    if isinstance(data, dict):
        parts = []
        if "question" in data:
            parts.append(f"Domanda: {data['question']}")
        if "answer" in data:
            parts.append(f"Risposta: {data['answer']}")
        if parts:
            return "\n\n".join(parts)
    # Fallback: pretty-print the whole structure
    return json.dumps(data, ensure_ascii=False, indent=2)
