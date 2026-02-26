import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib import cm

# Matikan log warning tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from audio_recorder_streamlit import audio_recorder
except ImportError:
    st.error("Library 'audio_recorder_streamlit' belum terinstall.")
    audio_recorder = None

# --- KONFIGURASI ---
MODEL_PATH = 'vocal_model_loso.h5' 
SR = 22050
DURATION = 3.0 
SAMPLES_PER_CHUNK = int(SR * DURATION)

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

# --- 2. PRE-PROCESSING & MAPPING (INTI DETEKSI) ---
def analyze_segment_by_segment(y, sr, model):
    results = []
    step = int(SR * 0.5)
    
    # Deteksi Pitch
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr
    )
    
    # Normalisasi dB range (Biar warnanya konsisten)
    # Suara biasanya range -80dB (hening) sampai 0dB (keras)
    MIN_DB = -80.0
    MAX_DB = 0.0
    
    for start in range(0, len(y) - SAMPLES_PER_CHUNK + 1, step):
        end = start + SAMPLES_PER_CHUNK
        chunk = y[start:end]
        
        # 1. Buat Mel-Spectrogram
        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # --- PERBAIKAN FATAL DI SINI (COLORMAP MAGMA) ---
        
        # A. Normalisasi nilai dB ke range 0.0 s.d 1.0
        # Supaya bisa diwarnai oleh Matplotlib
        mel_norm = (mel_db - MIN_DB) / (MAX_DB - MIN_DB)
        mel_norm = np.clip(mel_norm, 0, 1) # Pastikan gak ada yang minus
        
        # B. Terapkan Warna 'MAGMA' (Sama kayak Training)
        # cm.magma outputnya (128, 130, 4) -> RGBA
        colored_mel = cm.magma(mel_norm)
        
        # C. Ambil cuma RGB (3 Channel), buang Alpha
        input_rgb = colored_mel[..., :3] 
        
        # D. Tambah dimensi Batch -> (1, 128, 130, 3)
        input_data = input_rgb[np.newaxis, ...]
        
        # -----------------------------------------------
        
        # Prediksi
        pred = model.predict(input_data, verbose=0)
        score = float(pred[0][0])
        
        # Mapping Pitch
        f0_start = int(start / len(y) * len(f0))
        f0_end = int(end / len(y) * len(f0))
        chunk_f0 = f0[f0_start:f0_end]
        valid_freqs = chunk_f0[~np.isnan(chunk_f0)]
        
        if len(valid_freqs) > 0:
            avg_freq = np.median(valid_freqs)
            note = librosa.hz_to_note(avg_freq)
        else:
            avg_freq = 0.0
            note = "N/A"
            
        results.append({
            "Waktu (s)": round(start/sr, 2),
            "Freq (Hz)": round(avg_freq, 1),
            "Nada": note,
            "Skor AI": score,
            "mel_spec": mel_db # Simpan raw dB buat visualisasi matplotlib di bawah
        })
        
    return pd.DataFrame(results)

# --- 3. LOGIKA REKOMENDASI (CIRCLE OF FIFTHS) ---
def calculate_recommendation(df):
    df_clean = df[df['Nada'] != "N/A"].copy()
    
    if df_clean.empty:
        return "Audio Tidak Jelas", "Tidak Ada", "N/A", df
    
    ceiling_note = None
    min_score = 1.0
    
    # Cari Ceiling (Nada Tertinggi yang Resonant)
    for index, row in df_clean.iterrows():
        # Cari skor terendah
        if row['Skor AI'] < min_score:
            min_score = row['Skor AI']
            
        # Tessitura Logic: Update atap selama masih Resonant (> 0.5)
        if row['Skor AI'] > 0.5:
            ceiling_note = row['Nada']
            
    # Skenario 1: Hancur dari awal (Strained)
    if ceiling_note is None:
        first_note = df_clean.iloc[0]['Nada']
        return "Strained (Bahaya)", f"Turunkan dari {first_note}", first_note, df_clean
    
    # Skenario 2: Bagus semua (Resonant)
    if min_score > 0.5:
        last_note = df_clean.iloc[-1]['Nada']
        return "Resonant (Aman)", "C", last_note, df_clean

    # Skenario 3: Normal -> Hitung Mundur 7 Semitone (Perfect Fifth)
    try:
        ceiling_midi = librosa.note_to_midi(ceiling_note)
        
        # --- RUMUS SKRIPSI: Atap User = Nada Sol (5) ---
        # Maka Nada Dasar (Do) = Atap - 7 Semitone
        target_root_midi = ceiling_midi - 7 
        
        target_key = librosa.midi_to_note(target_root_midi)
        key_name = target_key[:-1] + " Mayor" # Format: C Mayor
        
        return "Batas Terdeteksi", key_name, ceiling_note, df_clean
        
    except:
        return "Error", "Cek Audio", ceiling_note, df_clean

# --- TAMPILAN UTAMA ---
def main():
    st.set_page_config(page_title="Vocal Profiler Pro", page_icon="🎤", layout="wide")
    
    st.title("🎤 Smart Vocal Profiler (Hz Analysis)")
    st.markdown("Analisis Frekuensi Fisik (Hz), Nada Musik, dan Kualitas Vokal AI.")

    # Sidebar Guide
    st.sidebar.header("⚙️ Setup & Panduan")
    gender = st.sidebar.radio("Mode:", ["Laki-laki (Start C3)", "Perempuan (Start C4)"])
    
    guide_file = "guide_male.wav" if "Laki" in gender else "guide_female.wav"
    
    if os.path.exists(guide_file):
        st.sidebar.write("🔊 **Contoh Nada Input:**")
        st.sidebar.audio(guide_file)
    else:
        st.sidebar.warning(f"File '{guide_file}' belum ada. Silakan rekam.")

    # Load Model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error Model: {e}")
        return

    # Input
    col1, col2 = st.columns(2)
    audio_data = None
    with col1:
        uploaded_file = st.file_uploader("Upload Audio (WAV/MP3)", type=["wav", "mp3"])
        if uploaded_file: audio_data, _ = librosa.load(uploaded_file, sr=SR)
    with col2:
        if audio_recorder:
            st.write("Klik mic untuk merekam:")
            audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=SR)
            if audio_bytes:
                with open("temp.wav", "wb") as f: f.write(audio_bytes)
                audio_data, _ = librosa.load("temp.wav", sr=SR)

    # Eksekusi
    if audio_data is not None:
        if st.button("🚀 MULAI SCANNING", type="primary"):
            with st.spinner('Menganalisis frekuensi & tekstur suara...'):
                
                df_result = analyze_segment_by_segment(audio_data, SR, model)
                
                if df_result.empty:
                    st.warning("Suara tidak terdeteksi.")
                    return

                status, rec_key, ceiling_note, df_final = calculate_recommendation(df_result)
                
                st.divider()
                
                # HASIL KIRI: KESIMPULAN
                c1, c2 = st.columns([1, 1.2])
                with c1:
                    st.subheader("🎹 Kesimpulan")
                    st.metric("Tessitura (Atap)", ceiling_note, help="Nada tertinggi yang nyaman")
                    st.info(f"Rekomendasi Nada Dasar: **{rec_key}**")
                    if "Strained" in status: st.error(status)
                    else: st.success(status)

                # HASIL KANAN: DATA MAPPING (TABEL)
                with c2:
                    st.subheader("📋 Data Mapping (Hz -> AI)")
                    st.caption("Detail frekuensi dan prediksi per segmen.")
                    
                    # Warna otomatis di tabel: Merah kalau Strained (<0.5)
                    def highlight_rows(val):
                        color = '#ffcdd2' if val < 0.5 else '#c8e6c9' # Merah muda / Hijau muda
                        return f'background-color: {color}'

                    # Tampilkan tabel tanpa kolom gambar (biar rapi)
                    display_table = df_final[['Waktu (s)', 'Freq (Hz)', 'Nada', 'Skor AI']].copy()
                    
                    st.dataframe(
                        display_table.style.applymap(
                            lambda x: 'color: red' if x < 0.5 else 'color: green', 
                            subset=['Skor AI']
                        ),
                        use_container_width=True,
                        height=250
                    )

                # GRAFIK VISUALISASI
                st.subheader("📊 Visualisasi Kualitas")
                colors = ['green' if s > 0.5 else 'red' for s in df_final['Skor AI']]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.bar(df_final['Nada'], df_final['Skor AI'], color=colors)
                
                # Tampilkan Hz di atas batang
                for i, bar in enumerate(bars):
                    freq_val = df_final.iloc[i]['Freq (Hz)']
                    if freq_val > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., 0.1,
                                f'{int(freq_val)}Hz',
                                ha='center', va='bottom', color='white', fontsize=8, rotation=90, fontweight='bold')

                ax.axhline(y=0.5, color='black', linestyle='--', label='Batas Aman')
                ax.set_ylabel("Kualitas Resonansi (1.0=Bagus)")
                plt.xticks(rotation=45)
                st.pyplot(fig)

if __name__ == "__main__":
    main()