import streamlit as st
import pandas as pd
import librosa
import os
from datetime import datetime
from streamlit_gsheets import GSheetsConnection
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

# --- LOGIKA PEMETAAN GITAR & CAPO ---
def get_guitar_advice(midi_val):
    pc = midi_val % 12
    guitar_families = [("C Major", 0, "C - G - Am - F"), ("G Major", 7, "G - D - Em - C"), ("D Major", 2, "D - A - Bm - G")]
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

# 7. PEMUTAR INSTRUMEN KUSTOM
st.markdown("---")
st.subheader("🎬 Yuk, Langsung Buktikan!")
rec_key_simple = rec_key_name.split()[0]
LAGU_LOKAL_DB = {"Sempurna - Andra and The Backbone": "sempurna", "Fix You - Coldplay": "fix_you"}
pilihan_lagu = st.selectbox("Pilih Lagu:", list(LAGU_LOKAL_DB.keys()))
file_target = f"assets/instrumentals/{LAGU_LOKAL_DB[pilihan_lagu]}_{rec_key_simple}.mp4"

if os.path.exists(file_target):
    st.video(file_target)
else:
    st.warning(f"Video untuk kunci {rec_key_simple} sedang disiapkan.")

# 8. FORMULIR DATA RESPONDEN (GOOGLE SHEETS & CSV)
st.markdown("---")
st.subheader("📋 Simpan Hasil & Download Data")
st.write("Silakan isi data untuk dikirim ke database riset kami.")

# Inisialisasi koneksi ke Google Sheets
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except:
    conn = None

with st.form("form_responden"):
    f_c1, f_c2 = st.columns(2)
    with f_c1:
        nama_res = st.text_input("Nama / Inisial")
        usia_res = st.number_input("Usia", 10, 70, 20)
        gender_res = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    with f_c2:
        tiktok_res = st.text_input("Username TikTok", placeholder="@username")
        tingkat_nyaman = st.select_slider("Kenyamanan kunci rekomendasi?", 
                                          options=["Tidak", "Kurang", "Cukup", "Nyaman", "Sangat Nyaman"], value="Nyaman")
        komentar_res = st.text_area("Masukan")
    
    submitted = st.form_submit_button("KIRIM DATA RISET 🚀", type="primary")

if submitted:
    if not nama_res:
        st.error("Nama wajib diisi.")
    else:
        # Menyiapkan baris data responden
        res_data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Nama": nama_res,
            "Usia": usia_res,
            "Gender": gender_res,
            "TikTok": tiktok_res,
            "Ceiling": ceiling_note,
            "Rec_Key": rec_key_name,
            "Capo": capo_fret,
            "Satisfaction": tingkat_nyaman,
            "Note": komentar_res,
            "AI_Score_Max": float(df_voices['ai_score'].max())
        }
        df_res = pd.DataFrame([res_data])

        # A. PROSES SIMPAN KE GOOGLE SHEETS
        if conn:
            try:
                existing_data = conn.read(worksheet="Sheet1", ttl=0)
                updated_df = pd.concat([existing_data, df_res], ignore_index=True)
                conn.update(worksheet="Sheet1", data=updated_df)
                st.success("✅ Data berhasil terkirim ke Google Sheets!")
            except Exception as e:
                st.error(f"Gagal kirim ke Google Sheets: {e}")

        # B. PROSES DOWNLOAD CSV (RESPON KHUSUS BARIS INI)
        st.info("💡 Kamu juga bisa mendownload data respondenmu di bawah ini sebagai bukti riset.")
        csv_res_only = df_res.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 DOWNLOAD DATA RESPONDEN (CSV)",
            data=csv_res_only,
            file_name=f"data_responden_{nama_res}.csv",
            mime='text/csv',
            use_container_width=True
        )
        st.balloons()

# 9. NAVIGASI AKHIR
st.divider()
col_a, col_b = st.columns(2)
with col_a:
    if st.button("🔄 Ulangi Sesi Baru"):
        st.session_state.clear()
        st.switch_page("app.py")
with col_b:
    # Backup: Download seluruh detail nada (bukan cuma data responden)
    csv_voices = df_voices.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Log Detil Nada (CSV)", data=csv_voices, 
                       file_name=f"log_vokal_{datetime.now().strftime('%H%M%S')}.csv",
                       use_container_width=True)