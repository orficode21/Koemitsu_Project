import streamlit as st
import pandas as pd
import librosa
import os
from src.ui_helpers import hero_box
from src.music_logic import key_name_from_root_midi, display_note_from_midi

st.set_page_config(page_title='Hasil Analisis - Koemitsu', layout='wide')

# 1. Proteksi Halaman
if 'step_results' not in st.session_state or not any(res is not None for res in st.session_state.step_results):
    st.error("Belum ada hasil analisis. Silakan selesaikan tes nada terlebih dahulu.")
    if st.button("Kembali ke Tes"): st.switch_page("pages/2_Test.py")
    st.stop()

hero_box('Langkah 3 — Hasil & Rekomendasi', 'Analisis Tessitura dan Rekomendasi Nada Dasar Musikal.')

# 2. Ambil Data 
results =[r for r in st.session_state.step_results if r is not None]
df = pd.DataFrame(results)

# 3. LOGIKA BERJENJANG MENCARI CEILING (ATAP AMAN)
threshold = float(st.session_state.get('res_threshold', 0.55))
ceiling_note = None
status_rekomendasi = ""
warna_status = "gray"

df_level1 = df[(df['pitch_ok'] == True) & (df['ai_score'] >= threshold)]

if not df_level1.empty:
    ceiling_note = df_level1.iloc[-1]['target_note']
    status_rekomendasi = "Optimal (Nada Pas & Suara Merdu)"
    warna_status = "green"
else:
    df_level2 = df[df['ai_score'] >= threshold]
    if not df_level2.empty:
        ceiling_note = df_level2.iloc[-1]['target_note']
        status_rekomendasi = "Estimasi (Berdasarkan Resonansi Suara)"
        warna_status = "blue"
    else:
        df_level3 = df[df['pitch_ok'] == True]
        if not df_level3.empty:
            ceiling_note = df_level3.iloc[-1]['target_note']
            status_rekomendasi = "Estimasi (Berdasarkan Ketepatan Nada)"
            warna_status = "orange"

if ceiling_note is None:
    ceiling_note = df.iloc[-1]['target_note']
    status_rekomendasi = "Batas Maksimal (Data Kurang Stabil)"
    warna_status = "red"

# 4. HITUNG REKOMENDASI KEY & FAMILY CHORD
ceiling_midi = int(librosa.note_to_midi(ceiling_note))
rec_root_midi = ceiling_midi - 7
rec_key_name = key_name_from_root_midi(rec_root_midi)
root_pc = rec_key_name.split()[0] # Inisial nada (misal: D)

# Logika Menghitung Family Chord Populer (I - V - vi - IV)
PITCH_CLASSES =['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
idx_root = rec_root_midi % 12
chord_I = PITCH_CLASSES[idx_root]
chord_V = PITCH_CLASSES[(idx_root + 7) % 12]
chord_vi = PITCH_CLASSES[(idx_root + 9) % 12] + "m" # Minor
chord_IV = PITCH_CLASSES[(idx_root + 5) % 12]
progresi_pop = f"{chord_I} - {chord_V} - {chord_vi} - {chord_IV}"

# Logika Menghitung Transpose dari Nada C
transpose_dari_C = idx_root
if transpose_dari_C > 6: 
    transpose_dari_C -= 12 # Agar tidak +11, jadikan -1

if transpose_dari_C == 0:
    transpose_text = "tidak perlu di-transpose (biarkan 0)"
elif transpose_dari_C > 0:
    transpose_text = f"tinggal di-transpose **+{transpose_dari_C}**"
else:
    transpose_text = f"tinggal di-transpose **{transpose_dari_C}**"

# 5. TAMPILAN DASHBOARD
st.markdown(f"### Status Analisis: :{warna_status}[{status_rekomendasi}]")

c1, c2, c3 = st.columns(3)
c1.metric("Atap Aman (Ceiling)", ceiling_note)
c2.metric("Rekomendasi Nada Dasar", rec_key_name)
c3.metric("Resonansi Tertinggi", f"{df['ai_score'].max():.2f}")

st.markdown("---")

# 6. VISUALISASI GRAFIK
st.write("### Grafik Kualitas Vokal Per Nada")
chart_data = df[['target_note_display', 'ai_score']].copy()
chart_data = chart_data.rename(columns={'target_note_display': 'Nada', 'ai_score': 'Skor Resonansi'})
st.line_chart(chart_data.set_index('Nada'))

# 7. KARTU KESIMPULAN & TIPS MUSIK
col_left, col_right = st.columns([1, 1.2])
with col_left:
    st.subheader("🎯 Kesimpulan")
    st.write(f"Nada **{ceiling_note}** adalah titik performa terbaikmu. Batas ini digunakan untuk menentukan nada dasar lagumu.")
    
    st.info(f"""
    💡 **Cara Cover Lagu (Transpose):**
    Karena nada dasarmu adalah **{rec_key_name}**, jika kamu ingin menyanyikan lagu yang aslinya bernada dasar **C Mayor** (chord C standar), kamu {transpose_text} pada gitar/keyboard/karaoke.
    """)

with col_right:
    st.subheader("🎸 Family Chord")
    st.success(f"Jika kamu menulis lagu atau bermain instrumen dengan nada dasar **{rec_key_name}**, ini adalah progresi chord yang paling umum dan pas untuk suaramu:")
    
    st.markdown(f"""
    **Progresi Populer (I - V - vi - IV):**  
    ### `{progresi_pop}`
    """)
    st.caption("Kombinasi Chord di atas adalah yang paling umum digunakan dan cocok untuk suaramu.")

# --- FITUR BARU: PEMUTAR INSTRUMEN LOKAL ---
st.markdown("---")
st.markdown("## 🎧 Ayo Langsung Nyanyi! (Tes Instrumen)")
st.write(f"Coba nyanyikan lagu di bawah ini. Instrumen ini sudah disesuaikan ke nada dasarmu (**{rec_key_name}**).")

# 1. Database Lagu Lokal (Nama Tampil : Nama File Asli tanpa ekstensi)
LAGU_LOKAL_DB = {
    "Sempurna - Andra and The Backbone": "sempurna",
    "Hati-Hati di Jalan - Tulus": "hati_hati_di_jalan",
    "Sial - Mahalini": "sial",
    "Fix You - Coldplay": "fix_you",
    "Kimi ga Kureta Mono - Secret Base": "secret_base"
}

pilihan_lagu = st.selectbox("Pilih Lagu Contoh:", list(LAGU_LOKAL_DB.keys()))
nama_file = LAGU_LOKAL_DB[pilihan_lagu]

# 2. Cek apakah file ada di folder
# Kita asumsikan file berekstensi .mp4 (video) atau .mp3 (audio)
path_folder = "assets/instrumentals"
path_mp4 = f"{path_folder}/{nama_file}.mp4"
path_mp3 = f"{path_folder}/{nama_file}.mp3"

with st.container(border=True):
    st.markdown(f"#### Memutar Instrumen: {pilihan_lagu}")
    
    if os.path.exists(path_mp4):
        st.video(path_mp4)
        st.success(f"✅ Video dimuat. Coba nyanyikan Reff-nya!")
    elif os.path.exists(path_mp3):
        st.audio(path_mp3)
        st.success(f"✅ Audio dimuat. Coba nyanyikan Reff-nya!")
    else:
        st.error(f"⚠️ File instrumen tidak ditemukan.")
        st.write(f"**Info untuk Developer (Kamu):** Pastikan kamu sudah meletakkan file bernama `{nama_file}.mp4` atau `{nama_file}.mp3` di dalam folder `{path_folder}/`.")

# 8. TABEL DETAIL
with st.expander("Lihat Data Mentah Analisis"):
    st.dataframe(df, use_container_width=True)

# 9. TOMBOL AKSI
st.divider()
if st.button("Ulangi Sesi Baru", type='primary'):
    st.session_state.clear()
    st.switch_page("app.py")