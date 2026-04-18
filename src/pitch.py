from __future__ import annotations

import math
from typing import Optional

import librosa
import numpy as np

from src.config import PYIN_FMAX, PYIN_FMIN, SR
from src.music_logic import display_note


def safe_voiced_f0(f0) -> np.ndarray:
    arr = np.asarray(f0, dtype=float)
    arr = arr[~np.isnan(arr)]
    arr = arr[arr > 0]
    return arr


def safe_median_f0(f0) -> Optional[float]:
    voiced = safe_voiced_f0(f0)
    if len(voiced) == 0:
        return None
    return float(np.median(voiced))


def detect_note_from_audio(y: np.ndarray, sr: int = SR) -> dict:
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=PYIN_FMIN,
        fmax=PYIN_FMAX,
        sr=sr,
    )
    median_hz = safe_median_f0(f0)
    voiced_ratio = float(np.mean(~np.isnan(f0))) if f0 is not None and len(f0) > 0 else 0.0
    confidence = float(np.nanmean(voiced_prob)) if voiced_prob is not None else None

    if median_hz is None:
        return {
            'hz': None,
            'note': None,
            'note_display': None,
            'midi': None,
            'voiced_ratio': voiced_ratio,
            'voiced_confidence': confidence,
        }

    note = librosa.hz_to_note(median_hz)
    midi = int(round(librosa.hz_to_midi(median_hz)))
    return {
        'hz': float(median_hz),
        'note': note,
        'note_display': display_note(note),
        'midi': midi,
        'voiced_ratio': voiced_ratio,
        'voiced_confidence': confidence,
    }


def cents_from_hz_to_target(detected_hz: Optional[float], target_note: str) -> Optional[float]:
    if detected_hz is None:
        return None
    target_hz = float(librosa.note_to_hz(target_note))
    if detected_hz <= 0 or target_hz <= 0:
        return None
    cents = 1200.0 * math.log2(detected_hz / target_hz)
    return float(cents)


def pitch_ok(cents_error: Optional[float], tolerance_cents: float = 150.0) -> bool:
    if cents_error is None:
        return False
    return abs(cents_error) <= tolerance_cents


def pitch_status_label(cents_error: Optional[float], tolerance_cents: float = 150.0) -> str:
    if cents_error is None:
        return 'Pitch tidak terbaca'
    abs_err = abs(cents_error)
    if abs_err <= tolerance_cents * 0.5:
        return 'Pas'
    if abs_err <= tolerance_cents:
        return 'Masih aman'
    return 'Meleset'
