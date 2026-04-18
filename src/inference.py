from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import streamlit as st
import tensorflow as tf

from src.config import MODEL_CANDIDATES, SAMPLES_SLICE_HOP, SAMPLES_SLICE_WIN, SR
from src.preprocessing import preprocess_chunk_to_rgb_magma

def find_model_path() -> Optional[Path]:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return None

@st.cache_resource(show_spinner=False)
def load_vocal_model():
    model_path = find_model_path()
    if model_path is None:
        return None
    try:
        # Load model tanpa compile untuk mempercepat dan menghindari error optimizers
        return tf.keras.models.load_model(str(model_path), compile=False)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

def score_resonance_for_note_segment(y_note, model, sr: int = SR) -> Tuple[Optional[float], Optional[np.ndarray]]:
    if model is None:
        return None, None
    
    scores = []
    last_spec_to_display = None # Untuk dikirim ke UI

    if len(y_note) < SAMPLES_SLICE_WIN:
        y_note = np.pad(y_note, (0, SAMPLES_SLICE_WIN - len(y_note)), mode='constant')

    stop = max(1, len(y_note) - SAMPLES_SLICE_WIN + 1)
    
    for start in range(0, stop, SAMPLES_SLICE_HOP):
        chunk = y_note[start:start + SAMPLES_SLICE_WIN]
        
        # BONGKAR TUPLE DI SINI (x_batch untuk AI, x_raw untuk Tampilan)
        x_batch, x_raw = preprocess_chunk_to_rgb_magma(chunk, sr=sr)
        
        # Sekarang x_batch hanya berisi 1 array tunggal, model tidak akan bingung lagi
        pred = model.predict(x_batch, verbose=0)
        
        scores.append(float(pred[0][0]))
        last_spec_to_display = x_raw 

    if not scores:
        return None, None
    
    return float(np.median(scores)), last_spec_to_display

def resonance_label(score: Optional[float], threshold: float) -> str:
    if score is None:
        return 'Belum dinilai AI'
    # Jika skor tinggi (misal 0.8) berarti Resonant
    return 'Resonant' if score >= threshold else 'Strained'