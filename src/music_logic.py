from __future__ import annotations

from dataclasses import dataclass
from typing import List

import librosa

from src.config import DEGREE_LABELS, DEGREE_SLUGS, DEGREE_TO_OFFSET, MAJOR_INTERVALS

NOTE_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
SHARP_TO_COMMON_FLAT = {'A#': 'Bb', 'D#': 'Eb', 'G#': 'Ab'}


@dataclass
class ScaleStep:
    index: int
    degree: str
    degree_slug: str
    target_midi: int
    target_note: str
    target_note_display: str


@dataclass
class SuggestedScale:
    root_midi: int
    root_note: str
    root_note_display: str
    key_name: str
    comfort_note: str
    comfort_note_display: str
    comfort_hz: float
    comfort_role: str
    steps: List[ScaleStep]


def to_flat_if_common(pitch_class: str) -> str:
    return SHARP_TO_COMMON_FLAT.get(pitch_class, pitch_class)


def midi_to_pitch_class(midi: int) -> str:
    return to_flat_if_common(NOTE_NAMES_SHARP[midi % 12])


def display_note(note: str) -> str:
    if not note:
        return note
    midi = int(librosa.note_to_midi(note))
    pc = midi_to_pitch_class(midi)
    octave = librosa.midi_to_note(midi, octave=True)[-1]
    if len(librosa.midi_to_note(midi, octave=True)) >= 3:
        octave = librosa.midi_to_note(midi, octave=True)[-1]
    return f'{pc}{octave}'


def display_note_from_midi(midi: int) -> str:
    return display_note(librosa.midi_to_note(midi, octave=True))


def build_major_scale_steps(root_midi: int) -> List[ScaleStep]:
    steps: List[ScaleStep] = []
    for idx, (degree, slug, interval) in enumerate(zip(DEGREE_LABELS, DEGREE_SLUGS, MAJOR_INTERVALS), start=1):
        target_midi = root_midi + interval
        target_note = librosa.midi_to_note(target_midi, octave=True)
        steps.append(
            ScaleStep(
                index=idx,
                degree=degree,
                degree_slug=slug,
                target_midi=target_midi,
                target_note=target_note,
                target_note_display=display_note(target_note),
            )
        )
    return steps

def key_name_from_root_midi(root_midi: int) -> str:
    return f"{midi_to_pitch_class(root_midi)} Mayor"

def suggest_root_from_comfort_midi(comfort_midi: int, comfort_role: str = 'Do') -> int:
    """
    Logika Baru (Mencari Atap Tessitura):
    Jadikan nada nyaman user langsung sebagai nada dasar (Do).
    Tangga nada akan bergerak naik dari titik nyaman ini.
    """
    offset = DEGREE_TO_OFFSET.get(comfort_role, 0)
    
    # Hitung nada dasar
    root_midi = comfort_midi - offset
    
    # PROTEKSI: Jika user bergumam terlalu rendah (di bawah G2 / MIDI 43),
    # kita naikkan 1 oktaf (+12) agar tesnya lebih menantang (mencari atap).
    # Tapi karena Ab2 adalah MIDI 44, dia tidak akan terpengaruh proteksi ini.
    if root_midi < 43:
        root_midi += 12
        
    return root_midi


def build_scale_from_comfort(comfort_note: str, comfort_hz: float, comfort_role: str = 'Do') -> SuggestedScale:
    comfort_midi = int(round(librosa.note_to_midi(comfort_note)))
    root_midi = suggest_root_from_comfort_midi(comfort_midi, comfort_role=comfort_role)
    root_note = librosa.midi_to_note(root_midi, octave=True)
    steps = build_major_scale_steps(root_midi)
    return SuggestedScale(
        root_midi=root_midi,
        root_note=root_note,
        root_note_display=display_note(root_note),
        key_name=key_name_from_root_midi(root_midi),
        comfort_note=comfort_note,
        comfort_note_display=display_note(comfort_note),
        comfort_hz=float(comfort_hz),
        comfort_role=comfort_role,
        steps=steps,
    )


def manual_root_options(center_root_midi: int, span: int = 5) -> list[tuple[str, int]]:
    options = []
    for midi in range(center_root_midi - span, center_root_midi + span + 1):
        note = librosa.midi_to_note(midi, octave=True)
        label = f"{display_note(note)} — {key_name_from_root_midi(midi)}"
        options.append((label, midi))
    return options


def step_targets_as_text(steps: List[ScaleStep]) -> str:
    return '  |  '.join(f"{step.degree} ({step.target_note_display})" for step in steps)
