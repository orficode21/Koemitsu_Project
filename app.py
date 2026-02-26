import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd

# Matikan log warning tensorflow biar terminal bersih
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Cek library mic
try:
    from audio_recorder_streamlit import audio_recorder
except ImportError:
    st.error("Library 'audio_recorder_streamlit' belum terinstall. Cek requirements.txt")
    audio_recorder = None

# --- KONFIGURASI SISTEM ---
MODEL_PATH = 'vocal_model_loso.h5' # Pastikan nama file ini BENAR
SR = 22050
DURATION = 3.0 
SAMPLES_PER_CHUNK = int(SR * DURATION)

# --- 1. LOAD MODEL AI ---
@st.cache_resource
def load_model():
    # compile=False biar ringan & menghindari warning optimizer
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

# --- 2. DETEKSI NADA (PITCH TRACKING) ---
def get_note_from_freq(freq):
    if freq == 0 or np.isnan(freq): return None
    return librosa.hz_to_note(freq)

def analyze_segment_by_segment(y, sr, model):
    """
    Memotong audio per langkah (Scanning), mendeteksi Pitch DAN Kualitas AI.
    Output: DataFrame berisi [Waktu, Nada, Skor AI]
    """
    results = []
    
    # Scanning dengan step 0.5 detik (Overlap agar detail)
    step = int(SR * 0.5)
    
    # Algoritma PYIN untuk deteksi nada vokal (C2-C6)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr
    )
    
    # Loop Scanning (Sliding Window)
    for start in range(0, len(y) - SAMPLES_PER_CHUNK + 1, step):
        end = start + SAMPLES_PER_CHUNK
        chunk = y[start:end]
        
        # A. PREDIKSI AI (Kualitas Suara)
        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Reshape & Duplikasi ke 3 Channel (RGB)
        # Input asli (128, 130) -> Jadi (1, 128, 130, 3)
        input_data = mel_db[np.newaxis, ..., np.newaxis]
        input_data = np.repeat(input_data, 3, axis=-1)
        
        pred = model.predict(input_data, verbose=0)
        score = float(pred[0][0]) # 0.0 (Strained) s.d 1.0 (Resonant)
        
        # B. DETEKSI PITCH (Nada rata-rata di segmen ini)
        # Mapping waktu audio ke frame pitch
        f0_start = int(start / len(y) * len(f0))
        f0_end = int(end / len(y) * len(f0))
        
        chunk_f0 = f0[f0_start:f0_end]
        valid_freqs = chunk_f0[~np.isnan(chunk_f0)]
        
        if len(valid_freqs) > 0:
            avg_freq = np.median(valid_freqs)
            note = librosa.hz_to_note(avg_freq)
        else:
            note = "N/A"
            
        results.append({
            "time": start/sr,
            "note": note,
            "score": score,
            "mel_spec": mel_db # Simpan untuk visualisasi
        })
        
    return pd.DataFrame(results)

# --- 3. LOGIKA MUSIK (CIRCLE OF FIFTHS) ---
def calculate_recommendation(df):
    """
    Menentukan Tessitura Ceiling dan Nada Dasar.
    """
    # Bersihkan data (hanya yang ada nadanya)
    df_clean = df[df['note'] != "N/A"].copy()
    
    if df_clean.empty:
        return "Audio Tidak Jelas", "Tidak Ada", "N/A", df
    
    # Variabel pelacak
    ceiling_note = None
    min_score = 1.0
    worst_note = "-"
    
    # Cari Ceiling: Nada terakhir yang skornya BAGUS (> 0.5)
    # Asumsi user menyanyi scale NAIK (Rendah -> Tinggi)
    for index, row in df_clean.iterrows():
        # Lacak skor terburuk untuk status
        if row['score'] < min_score:
            min_score = row['score']
            worst_note = row['note']
            
        # Update ceiling selama masih Resonant
        if row['score'] > 0.5:
            ceiling_note = row['note']
            
    # Skenario 1: Suara Hancur Semua (Strained dari awal)
    if ceiling_note is None:
        first_note = df_clean.iloc[0]['note']
        return "Strained (Bahaya)", f"Turunkan dari {first_note}", first_note, df_clean
    
    # Skenario 2: Suara Bagus Semua (Belum ketemu batas)
    if min_score > 0.5:
        last_note = df_clean.iloc[-1]['note']
        return "Resonant (Aman)", "Nada Asli (Original)", last_note, df_clean

    # Skenario 3: Normal (Ketemu Batas/Ceiling) -> Hitung Circle of Fifths
    # Rumus: Nada Dasar = Ceiling Note - 7 Semitone (Perfect Fifth)
    try:
        ceiling_midi = librosa.note_to_midi(ceiling_note)
        target_root_midi = ceiling_midi - 7 
        target_key = librosa.midi_to_note(target_root_midi)
        
        # Format teks kunci (misal C4 -> C Mayor)
        key_name = target_key[:-1] + " Mayor"
        return "Batas Terdeteksi", key_name, ceiling_note, df_clean
        
    except:
        return "Error Perhitungan", "Cek Audio", ceiling_note, df_clean

# --- TAMPILAN UTAMA (FRONTEND) ---
def main():
    st.set_page_config(page_title="Smart Vocal Profiler", page_icon="🎤", layout="wide")
    
    st.title("🎤 Smart Vocal Profiler & Transposer")
    st.markdown("Sistem AI untuk mendeteksi batas aman nada (**Tessitura**) dan rekomendasi musik.")

    # --- SIDEBAR: PENGATURAN & AUDIO GUIDE ---
    st.sidebar.header("⚙️ Pengaturan & Panduan")
    gender = st.sidebar.radio("Jenis Suara Anda:", ["Laki-laki (Low)", "Perempuan (High)"])
    
    if gender == "Laki-laki (Low)":
        start_note = "C3"
        guide_file = "guide_male.wav"
    else:
        start_note = "C4"
        guide_file = "guide_female.wav"
        
    st.sidebar.info(f"💡 **Instruksi:**\nMulailah menyanyi vokal **'AAAA'** dari nada rendah (**{start_note}**), lalu naik terus setinggi yang Anda bisa sampai suara terasa tegang.")
    
    st.sidebar.write("🔊 **Contoh Nada Input:**")
    if os.path.exists(guide_file):
        st.sidebar.audio(guide_file)
    else:
        st.sidebar.warning(f"File '{guide_file}' belum ada di folder.")

    # Load Model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"❌ Gagal memuat model! Pastikan file '{MODEL_PATH}' ada.\nError: {e}")
        return

    # --- INPUT USER ---
    col1, col2 = st.columns(2)
    audio_data = None
    
    with col1:
        st.subheader("📂 Upload File")
        uploaded_file = st.file_uploader("Format WAV/MP3", type=["wav", "mp3"])
        if uploaded_file:
            audio_data, _ = librosa.load(uploaded_file, sr=SR)
            
    with col2:
        st.subheader("🎙️ Rekam Langsung")
        if audio_recorder is not None:
            audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=SR, text="Klik mic untuk merekam")
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                with open("temp_mic.wav", "wb") as f: f.write(audio_bytes)
                audio_data, _ = librosa.load("temp_mic.wav", sr=SR)

    # --- EKSEKUSI ---
    if audio_data is not None:
        if st.button("🚀 MULAI ANALISIS", type="primary"):
            with st.spinner('AI sedang memindai tessitura Anda...'):
                
                # 1. Analisis
                df_result = analyze_segment_by_segment(audio_data, SR, model)
                
                if df_result.empty:
                    st.error("Audio terlalu pendek atau hening. Coba lagi.")
                    return

                # 2. Hitung Rekomendasi
                status, rec_key, ceiling_note, df_final = calculate_recommendation(df_result)
                
                # --- TAMPILAN HASIL ---
                st.divider()
                
                # BAGIAN KIRI: REKOMENDASI
                left_col, right_col = st.columns([1, 2])
                
                with left_col:
                    st.subheader("🎹 Hasil Diagnosa")
                    st.metric("Tessitura (Batas Aman)", ceiling_note)
                    
                    if "Aman" in status:
                        st.success(f"✅ {status}")
                    elif "Strained" in status:
                        st.error(f"❌ {status}")
                    else:
                        st.warning(f"⚠️ {status}")
                        
                    st.info(f"🎼 Rekomendasi Nada Dasar:\n### **{rec_key}**")
                    st.caption("Metode: Circle of Fifths Transposition")

                # BAGIAN KANAN: GRAFIK & VISUALISASI
                with right_col:
                    st.subheader("📊 Grafik Kualitas per Nada")
                    
                    # Warna Bar: Hijau (>0.5) / Merah (<0.5)
                    colors = ['#28a745' if s > 0.5 else '#dc3545' for s in df_final['score']]
                    
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.bar(df_final['note'], df_final['score'], color=colors)
                    ax.axhline(y=0.5, color='gray', linestyle='--', label='Batas Aman')
                    ax.set_ylim(0, 1.1)
                    ax.set_ylabel("Skor Resonansi (1.0=Bagus)")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
                    # Tampilkan Spektrogram Terburuk (Bukti Visual)
                    worst_idx = df_final['score'].idxmin()
                    worst_spec = df_final.loc[worst_idx, 'mel_spec']
                    worst_note = df_final.loc[worst_idx, 'note']
                    worst_score = df_final.loc[worst_idx, 'score']
                    
                    st.write(f"**🔍 Bukti Visual (Strained di nada {worst_note}):**")
                    fig2, ax2 = plt.subplots(figsize=(8, 2))
                    img = librosa.display.specshow(worst_spec, sr=SR, x_axis='time', y_axis='mel', ax=ax2)
                    fig2.colorbar(img, ax=ax2, format='%+2.0f dB')
                    st.pyplot(fig2)

if __name__ == "__main__":
    main()