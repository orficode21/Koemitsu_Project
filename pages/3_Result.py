import streamlit as st
import pandas as pd
import librosa
import os
from datetime import datetime
from src.ui_helpers import hero_box
from src.music_logic import key_name_from_root_midi, display_note_from_midi

st.set_page_config(page_title='Hasil Analisis - Koemitsu', layout='wide')

# 1. Proteksi Halaman
if 'step_results' not in st.session_state or not any(res is not None for res in st.session_state.step_results):
    st.error("Belum ada hasil analisis. Silakan selesaikan tes nada terlebih dahulu.")
    if st.button("Kembali ke Tes"): st.switch_page("pages/2_Test.py")
    st.stop()

hero_box('Langkah 3 — Hasil & Rekomendasi', 'AI telah memetakan Tessitura dan Rekomendasi Nada Dasarmu.')

# 2. Ambil Data Analisis
results = [r for r in st.session_state.step_results if r is not None]
df = pd.DataFrame(results)

# 3. LOGIKA BERJENJANG MENCARI CEILING (ATAP AMAN)
threshold = float(st.session_state.get('res_threshold', 0.55))
ceiling_note = None
status_rekomendasi = ""
warna_status = "gray"

df_level1 = df[(df['pitch_ok'] == True) & (df['ai_score'] >= threshold)]

if not df_level1.empty:
    ceiling_note = df_level1.iloc[-1]['target_note']
    status_rekomendasi = "Optimal (Nada Pas & Resonant)"
    warna_status = "green"
else:
    df_level2 = df[df['ai_score'] >= threshold]
    if not df_level2.empty:
        ceiling_note = df_level2.iloc[-1]['target_note']
        status_rekomendasi = "Estimasi (Berdasarkan Resonansi)"
        warna_status = "blue"
    else:
        df_level3 = df[df['pitch_ok'] == True]
        if not df_level3.empty:
            ceiling_note = df_level3.iloc[-1]['target_note']
            status_rekomendasi = "Estimasi (Berdasarkan Pitch)"
            warna_status = "orange"

if ceiling_note is None:
    ceiling_note = df.iloc[-1]['target_note']
    status_rekomendasi = "Batas Maksimal (Data Kurang Stabil)"
    warna_status = "red"

# 4. HITUNG REKOMENDASI KEY & FAMILY CHORD
ceiling_midi = int(librosa.note_to_midi(ceiling_note))
rec_root_midi = ceiling_midi - 7
rec_key_name = key_name_from_root_midi(rec_root_midi)

# Logika Family Chord (I - V - vi - IV)
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
idx_root = rec_root_midi % 12
chord_I = PITCH_CLASSES[idx_root]
chord_V = PITCH_CLASSES[(idx_root + 7) % 12]
chord_vi = PITCH_CLASSES[(idx_root + 9) % 12] + "m"
chord_IV = PITCH_CLASSES[(idx_root + 5) % 12]
progresi_pop = f"{chord_I} - {chord_V} - {chord_vi} - {chord_IV}"

# 5. TAMPILAN DASHBOARD
st.markdown(f"### Status Analisis: :{warna_status}[{status_rekomendasi}]")
c1, c2, c3 = st.columns(3)
c1.metric("Atap Aman (Ceiling)", ceiling_note)
c2.metric("Nada Dasar Utama", rec_key_name)
c3.metric("Skor Resonansi Tertinggi", f"{df['ai_score'].max():.2f}")

st.markdown("---")

# 6. VISUALISASI & TIPS
col_left, col_right = st.columns([1, 1.2])
with col_left:
    st.subheader("🎯 Kesimpulan")
    st.write(f"Batas atas kualitas suaramu berada di nada **{ceiling_note}**.")
    st.info(f"💡 **Tips:** Jika lagu asli menggunakan kunci **C Major**, kamu cukup transpose **{rec_root_midi%12 if rec_root_midi%12 <=6 else rec_root_midi%12 - 12:+}** pada alat musikmu.")
    
    st.write("### Grafik Stabilitas Vokal")
    chart_data = df[['target_note_display', 'ai_score']].copy()
    chart_data = chart_data.rename(columns={'target_note_display': 'Nada', 'ai_score': 'Skor Resonansi'})
    st.line_chart(chart_data.set_index('Nada'))

with col_right:
    st.subheader("🎸 Family Chord (Keluarga Kunci)")
    st.success(f"Gunakan nada dasar **{rec_key_name}** dengan progresi chord pop ini:")
    st.markdown(f"## `{progresi_pop}`")
    st.caption("Kombinasi ini akan menjamin puncak lagu tetap berada di wilayah nyamanmu.")

# 7. PEMUTAR INSTRUMEN LOKAL
st.markdown("---")
st.subheader("🎧 Ayo Langsung Nyanyi! (Cek Hasil)")
LAGU_LOKAL_DB = {
    "Sempurna - Andra and The Backbone": "sempurna",
    "Hati-Hati di Jalan - Tulus": "hati_hati_di_jalan",
    "Sial - Mahalini": "sial",
    "Fix You - Coldplay": "fix_you",
    "Kimi ga Kureta Mono - Secret Base": "secret_base"
}
pilihan_lagu = st.selectbox("Pilih Lagu Contoh:", list(LAGU_LOKAL_DB.keys()))
nama_file = LAGU_LOKAL_DB[pilihan_lagu]
path_folder = "assets/instrumentals"

with st.container(border=True):
    # Cek file mp4 atau mp3
    if os.path.exists(f"{path_folder}/{nama_file}.mp4"):
        st.video(f"{path_folder}/{nama_file}.mp4")
    elif os.path.exists(f"{path_folder}/{nama_file}.mp3"):
        st.audio(f"{path_folder}/{nama_file}.mp3")
    else:
        st.warning(f"File instrumen `{nama_file}` belum ada di folder `{path_folder}`.")

# 8. FORM DATA RESPONDEN (PENTING UNTUK SKRIPSI)
st.markdown("---")
st.subheader("📋 Formulir Responden")
st.write("Mohon isi data di bawah ini untuk membantu penelitian saya.")

with st.form("form_responden"):
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        nama = st.text_input("Nama Lengkap / Inisial")
        usia = st.number_input("Usia", min_value=10, max_value=70, value=20)
    with col_f2:
        gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        tingkat_nyaman = st.select_slider("Seberapa nyaman kunci rekomendasi tadi?", options=["Sangat Tidak Nyaman", "Kurang Nyaman", "Cukup", "Nyaman", "Sangat Nyaman"], value="Nyaman")
    
    komentar = st.text_area("Komentar tambahan (opsional)")
    
    submitted = st.form_submit_button("Submit Data Riset", type="primary")

if submitted:
    if not nama:
        st.error("Mohon isi nama Anda.")
    else:
        # Siapkan data untuk disimpan
        row_data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Nama": nama,
            "Usia": usia,
            "Gender": gender,
            "Ceiling": ceiling_note,
            "Rec_Key": rec_key_name,
            "AI_Max_Score": df['ai_score'].max(),
            "Kepuasan_User": tingkat_nyaman,
            "Komentar": komentar
        }
        
        # Simpan ke CSV lokal (Akan tersimpan di server selama sesi berjalan)
        log_file = "data/respondent_logs.csv"
        log_df = pd.DataFrame([row_data])
        
        if not os.path.isfile(log_file):
            log_df.to_csv(log_file, index=False)
        else:
            log_df.to_csv(log_file, mode='a', header=False, index=False)
            
        st.success(f"Terima kasih {nama}! Datamu telah tersimpan untuk riset.")
        st.balloons()

# 9. NAVIGASI AKHIR
st.divider()
col_a, col_b = st.columns(2)
with col_a:
    if st.button("Ulangi Sesi Baru"):
        st.session_state.clear()
        st.switch_page("app.py")
with col_b:
    # Tombol download untuk jaga-jaga kalau file CSV di server hilang
    csv_download = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Detail Log Suara (CSV)", data=csv_download, file_name=f"log_{datetime.now().strftime('%H%M%S')}.csv")