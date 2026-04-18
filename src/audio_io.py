from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from src.config import (
    AUDIO_FILE_TYPES,
    CALIBRATION_GUIDE_CANDIDATES,
    GUIDE_DEGREES_DIR,
    GUIDE_NOTES_DIR,
    GUIDE_SAMPLE_RATE,
    GUIDE_TONE_DURATION,
    GUIDE_TONE_GAIN,
    SR,
)


def get_uploaded_bytes(uploaded_file) -> Optional[bytes]:
    if uploaded_file is None:
        return None
    if hasattr(uploaded_file, 'getvalue'):
        return uploaded_file.getvalue()
    try:
        return uploaded_file.read()
    except Exception:
        return None


def infer_suffix(uploaded_file, default: str = '.wav') -> str:
    if uploaded_file is None:
        return default
    name = getattr(uploaded_file, 'name', '') or ''
    suffix = Path(name).suffix.lower()
    return suffix if suffix else default


def load_audio_from_bytes(audio_bytes: bytes, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
    return y


def load_audio_source(uploaded_file, sr: int = SR) -> Tuple[Optional[bytes], Optional[np.ndarray], str]:
    audio_bytes = get_uploaded_bytes(uploaded_file)
    if audio_bytes is None:
        return None, None, '.wav'
    suffix = infer_suffix(uploaded_file)
    y = load_audio_from_bytes(audio_bytes, sr=sr)
    return audio_bytes, y, suffix


def save_audio_bytes(audio_bytes: bytes, path: Path, sr: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == '.wav':
        data = load_audio_from_bytes(audio_bytes, sr=sr)
        sf.write(str(path), data, sr)
    else:
        path.write_bytes(audio_bytes)
    return path


def note_filename_candidates(note: str) -> list[str]:
    safe = note.replace('#', 'sharp').replace('b', 'flat')
    return [
        f'{note}.wav',
        f'{safe}.wav',
        f'{note.lower()}.wav',
        f'{safe.lower()}.wav',
    ]


def degree_filename_candidates(degree_slug: str) -> list[str]:
    return [f'{degree_slug}.wav']


def find_custom_note_guide(note: str, degree_slug: Optional[str] = None) -> Optional[Path]:
    for filename in note_filename_candidates(note):
        candidate = GUIDE_NOTES_DIR / filename
        if candidate.exists():
            return candidate
    if degree_slug:
        for filename in degree_filename_candidates(degree_slug):
            candidate = GUIDE_DEGREES_DIR / filename
            if candidate.exists():
                return candidate
    return None


def find_calibration_guide() -> Optional[Path]:
    for path in CALIBRATION_GUIDE_CANDIDATES:
        if path.exists():
            return path
    return None


def synthesize_note_tone(note: str, duration: float = GUIDE_TONE_DURATION, sr: int = GUIDE_SAMPLE_RATE) -> bytes:
    hz = float(librosa.note_to_hz(note))
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = GUIDE_TONE_GAIN * np.sin(2 * np.pi * hz * t)
    fade_len = int(0.03 * sr)
    if fade_len > 0 and len(audio) > fade_len * 2:
        fade = np.linspace(0.0, 1.0, fade_len)
        audio[:fade_len] *= fade
        audio[-fade_len:] *= fade[::-1]
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    buffer.seek(0)
    return buffer.read()


def get_guide_audio_bytes(note: str, degree_slug: Optional[str] = None) -> tuple[Optional[bytes], str]:
    custom = find_custom_note_guide(note, degree_slug=degree_slug)
    if custom and custom.exists():
        return custom.read_bytes(), 'custom'
    return synthesize_note_tone(note), 'generated'


def allowed_audio_help_text() -> str:
    return 'Format yang didukung: ' + ', '.join(ext.upper() for ext in AUDIO_FILE_TYPES)
