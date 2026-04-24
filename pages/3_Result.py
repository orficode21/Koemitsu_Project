import streamlit as st
import pandas as pd
import librosa
import os
from datetime import datetime
from src.ui_helpers import hero_box
from src.music_logic import key_name_from_root_midi, display_note_from_midi

# -- KONFIGURASI HALAMAN --
st.set_page_config(page_title='Hasil Analisis - Koemitsu', layout='wide')

# 1. Proteksi Halaman
if 'step_results' not in st.session_state or not any(res is not None for res in st.session_state.step_results):
    st.error("Belum ada hasil analisis. Silakan selesaikan tes nada terlebih dahulu.")
    if st.button("Kembali ke Tes"): 
        st.switch_page("pages/2_Test.py")
    st.stop()

hero_box('Langkah 3 — Hasil & Rekomendasi', 'AI telah memetakan wilayah suara ternyamanmu.')

# 2. Ambil Data Analisis Suara
results = [r for r in st.session_state.step_results if r is not None]
df_voices = pd.DataFrame(results)

# 3. LOGIKA BERJENJANG MENCARI CEILING (ATAP AMAN)
threshold = float(st.session_state.get('res_threshold', 0.55))
ceiling_note = None
status_rekomendasi = ""
warna_status = "gray"

df_level1 = df_voices[(df_voices['pitch_ok'] == True) & (df_voices['ai_score'] >= threshold)]

if not df_level1.empty:
    ceiling_note = df_level1.iloc[-1]['target_note']
    status_rekomendasi = "Optimal (Nada Pas & Merdu)"
    warna_status = "green"
else:
    df_level2 = df_voices[df_voices['ai_score'] >= threshold]
    if not df_level2.empty:
        ceiling_note = df_level2.iloc[-1]['target_note']
        status_rekomendasi = "Estimasi (Berdasarkan Resonansi)"
        warna_status = "blue"
    else:
        df_level3 = df_voices[df_voices['pitch_ok'] == True]
        if not df_level3.empty:
            ceiling_note = df_level3.iloc[-1]['target_note']
            status_rekomendasi = "Estimasi (Berdasarkan Pitch)"
            warna_status = "orange"

if ceiling_note is None:
    ceiling_note = df_voices.iloc[-1]['target_note']
    status_rekomendasi = "Batas Maksimal (Data Kurang Stabil)"
    warna_status = "red"

# 4. HITUNG REKOMENDASI KEY & LOGIKA TRANSPOSISI
ceiling_midi = int(librosa.note_to_midi(ceiling_note))
rec_root_midi = ceiling_midi - 7
rec_key_name = key_name_from_root_midi(rec_root_midi)
target_pc = rec_root_midi % 12 

# Array PITCH_CLASSES mutlak pakai Sharp (#) untuk nama file
PITCH_CLASSES_SHARP =['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Mengambil inisial nada HANYA dengan format # (C#, D#, dll) untuk mencari file
file_key_rekomendasi = PITCH_CLASSES_SHARP[target_pc]

# --- LOGIKA PEMETAAN GITAR & CAPO ---
def get_guitar_advice(midi_val):
    pc = midi_val % 12
    guitar_families =[("C Major", 0, "C - G - Am - F"), ("G Major", 7, "G - D - Em - C"), ("D Major", 2, "D - A - Bm - G")]
    best_fret, best_family, best_prog = 13, "", ""
    for name, root, prog in guitar_families:
        fret = (pc - root) % 12
        if fret < best_fret:
            best_fret, best_family, best_prog = fret, name, prog
    return best_fret, best_family, best_prog

capo_fret, family_name, family_prog = get_guitar_advice(target_pc)
transpose_val = target_pc if target_pc <= 6 else target_pc - 12

# 5. TAMPILAN DASHBOARD
st.markdown(f"### Status Vokal: :{warna_status}[{status_rekomendasi}]")
c1, c2, c3 = st.columns(3)
c1.metric("Atap Aman", ceiling_note)
c2.metric("Rekomendasi Kunci", rec_key_name)
c3.metric("Stabilitas Maksimal", f"{int(df_voices['ai_score'].max()*100)}%")

st.markdown("---")

# 6. PETUNJUK SETTING
st.subheader("🎵 Cara Mengatur Nada Dasar Lagu")
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.info(f"**Keyboard / Karaoke:**\nJika lagu di kunci C, Transpose: **{transpose_val:+}**")
    st.success(f"**Pemain Gitar:**\nPakai Chord **{family_name.split()[0]}**, Capo Fret: **{capo_fret}**")

with col_right:
    st.markdown(f"**Contoh Progresi Chord ({rec_key_name}):**")
    st.markdown(f"### `{family_prog}`")
    chart_data = df_voices[['target_note_display', 'ai_score']].copy()
    st.line_chart(chart_data.rename(columns={'ai_score': 'Skor'}).set_index('target_note_display'), height=150)

# 7. PEMUTAR INSTRUMEN KUSTOM (A/B TESTING)
st.markdown("---")
st.subheader("🎬 Buktikan Bedanya! (A/B Testing)")
st.write("Silakan nyanyikan bagian Reff pada kunci asli, lalu rasakan perbedaannya saat menyanyi dengan kunci rekomendasi AI.")

# Database Lagu dan Original Key-nya (Pasti pakai #)
LAGU_LOKAL_DB = {
    "Sempurna - Andra and The Backbone": {"slug": "sempurna", "ori_key": "E"},
    "Fix You - Coldplay": {"slug": "fix_you", "ori_key": "D#"}
}

pilihan_lagu = st.selectbox("Pilih Lagu untuk Dicoba:", list(LAGU_LOKAL_DB.keys()))
slug_lagu = LAGU_LOKAL_DB[pilihan_lagu]["slug"]
ori_key_lagu = LAGU_LOKAL_DB[pilihan_lagu]["ori_key"]
path_folder = "assets/instrumentals"

# File Paths (Otomatis menyesuaikan penamaan C#, D#, G# dst)
file_original = f"{path_folder}/{slug_lagu}_{ori_key_lagu}.mp4"
file_rekomendasi = f"{path_folder}/{slug_lagu}_{file_key_rekomendasi}.mp4"

col_vid1, col_vid2 = st.columns(2)

# Video Kunci Asli
with col_vid1:
    with st.container(border=True):
        st.markdown(f"#### 1️⃣ Kunci Asli ({ori_key_lagu} Major)")
        st.caption("Coba nyanyikan ini dulu.")
        if os.path.exists(file_original):
            st.video(file_original)
        else:
            st.error(f"File `{slug_lagu}_{ori_key_lagu}.mp4` tidak ditemukan.")

# Video Rekomendasi AI
with col_vid2:
    with st.container(border=True):
        st.markdown(f"#### 2️⃣ Kunci AI ({rec_key_name})") # Teks tetap pakai yg cantik (bisa Eb atau D#)
        st.caption("Rasakan bedanya di tenggorokanmu!")
        if os.path.exists(file_rekomendasi):
            st.video(file_rekomendasi)
        else:
            st.error(f"File `{slug_lagu}_{file_key_rekomendasi}.mp4` sedang disiapkan.")

# 8. KUESIONER GOOGLE FORM (TOMBOL BESAR)
st.markdown("---")
st.subheader("📋 Isi Kuesioner Riset")
st.write("Setelah mencoba kedua video di atas, mohon isi kuesioner singkat ini untuk data penelitian skripsi.")

# Ubah LINK_GOOGLE_FORM_KAMU dengan link G-Form yang asli
st.link_button("📝 KLIK DI SINI UNTUK MENGISI KUESIONER PENELITIAN", "https://forms.gle/cdLoK1ehPgNGQG838", type="primary", use_container_width=True)

# 9. NAVIGASI AKHIR
st.divider()
col_a, col_b = st.columns(2)
with col_a:
    if st.button("🔄 Ulangi Sesi Baru"):
        st.session_state.clear()
        st.switch_page("app.py") 
with col_b:
    csv_voices = df_voices.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Log Detil Nada (CSV)", data=csv_voices, 
                       file_name=f"log_vokal_{datetime.now().strftime('%H%M%S')}.csv",
                       use_container_width=True)