from __future__ import annotations
from typing import Any, Optional
import pandas as pd
import streamlit as st # PENTING: Harus diimport untuk simpan gambar ke session_state

from src.config import DEFAULT_PITCH_TOLERANCE_CENTS
from src.inference import resonance_label, score_resonance_for_note_segment
from src.music_logic import build_scale_from_comfort, display_note
from src.pitch import cents_from_hz_to_target, detect_note_from_audio, pitch_ok, pitch_status_label

def analyze_calibration_take(y, comfort_role: str = 'Do') -> dict[str, Any]:
    """Proses pengambilan nada nyaman awal (Anchor)."""
    pitch = detect_note_from_audio(y)
    if pitch['note'] is None:
        return {
            'ok': False,
            'message': 'Pitch kalibrasi belum terbaca. Coba tahan vokal “aaa” lebih stabil 2–3 detik.',
        }
    suggestion = build_scale_from_comfort(
        comfort_note=pitch['note'],
        comfort_hz=pitch['hz'],
        comfort_role=comfort_role,
    )
    return {
        'ok': True,
        'pitch': pitch,
        'suggestion': suggestion,
    }

def analyze_note_take(
    y,
    target_note: str,
    target_note_display: str,
    degree: str,
    model,
    res_threshold: float,
    tolerance_cents: float = DEFAULT_PITCH_TOLERANCE_CENTS,
) -> dict[str, Any]:
    """Proses analisis per nada (Do-Re-Mi)."""
    
    # 1. Analisis Pitch menggunakan pYIN
    pitch = detect_note_from_audio(y)
    cents_error = cents_from_hz_to_target(pitch['hz'], target_note)
    is_pitch_ok = pitch_ok(cents_error, tolerance_cents=tolerance_cents)
    
    # 2. Analisis Kualitas menggunakan CNN
    # Menangkap 2 return: (score, image) untuk menghindari ValueError cardinality
    ai_score, spec_img = score_resonance_for_note_segment(y, model)
    
    quality = resonance_label(ai_score, threshold=res_threshold)

    # 3. Simpan gambar spektrogram ke session_state agar bisa dipanggil di UI pages/2_Test.py
    if spec_img is not None:
        st.session_state[f"last_spec_img_{degree}"] = spec_img

    # 4. Kembalikan hasil dalam bentuk dictionary
    return {
        'degree': degree,
        'target_note': target_note,
        'target_note_display': target_note_display,
        'detected_note': pitch['note'],
        'detected_note_display': pitch['note_display'],
        'detected_hz': None if pitch['hz'] is None else round(float(pitch['hz']), 2),
        'voiced_ratio': round(float(pitch['voiced_ratio']), 3),
        'voiced_confidence': None if pitch['voiced_confidence'] is None else round(float(pitch['voiced_confidence']), 3),
        'cents_error': None if cents_error is None else round(float(cents_error), 1),
        'pitch_ok': bool(is_pitch_ok),
        'pitch_status': pitch_status_label(cents_error, tolerance_cents=tolerance_cents),
        'ai_score': None if ai_score is None else round(float(ai_score), 4),
        'quality': quality,
    }

def summarize_results(results: list[dict[str, Any]], root_midi: int, key_name: str, comfort_note_display: str, res_threshold: float) -> dict[str, Any]:
    """Merekap seluruh hasil tes untuk halaman Result."""
    if not results:
        return {
            'completed_steps': 0,
            'safe_ceiling': None,
            'safe_ceiling_display': '-',
            'pitch_pass_count': 0,
            'resonant_count': 0,
            'key_name': key_name,
            'comfort_note_display': comfort_note_display,
        }

    df = pd.DataFrame(results)
    pitch_pass_df = df[df['pitch_ok'] == True].copy()
    
    # Filter nada yang pas pitch-nya DAN skor AI-nya di atas ambang batas
    if 'ai_score' in df.columns:
        usable_df = df[(df['pitch_ok'] == True) & (df['ai_score'] >= res_threshold)].copy()
    else:
        usable_df = pitch_pass_df

    safe_ceiling = None
    if not usable_df.empty:
        # Ambil nada tertinggi yang lolos (baris paling akhir)
        safe_ceiling = usable_df.iloc[-1]['target_note']

    resonant_count = 0
    if 'quality' in df.columns:
        resonant_count = int((df['quality'] == 'Resonant').sum())

    return {
        'completed_steps': int(len(df)),
        'safe_ceiling': safe_ceiling,
        'safe_ceiling_display': display_note(safe_ceiling) if safe_ceiling else '-',
        'pitch_pass_count': int(len(pitch_pass_df)),
        'resonant_count': resonant_count,
        'key_name': key_name,
        'comfort_note_display': comfort_note_display,
    }

def results_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Mengonversi list dictionary ke DataFrame untuk tabel Streamlit."""
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    # Susun urutan kolom agar rapi saat ditampilkan di tabel
    order = [
        'degree',
        'target_note_display',
        'detected_note_display',
        'detected_hz',
        'cents_error',
        'pitch_ok',
        'pitch_status',
        'ai_score',
        'quality',
    ]
    # Pastikan hanya kolom yang ada di DataFrame yang diambil
    cols = [c for c in order if c in df.columns]
    return df[cols]