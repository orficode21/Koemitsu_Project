import streamlit as st
from src.audio_io import load_audio_source
from src.analysis import analyze_calibration_take
from src.ui_helpers import hero_box, card_open, card_close, render_audio_capture, render_calibration_guide, render_audio_settings_sidebar

st.set_page_config(page_title='Kalibrasi - Koemitsu', layout='wide')

# ==========================================
# 🛡️ SATPAM SESI (Mencegah Error saat Refresh)
# ==========================================
if 'session_id' not in st.session_state:
    st.warning("🔄 Sesi kamu telah berakhir atau halaman di-refresh. Untuk menjaga keakuratan data, silakan mulai kembali dari halaman utama ya!")
    if st.button("🏠 Kembali ke Beranda", type="primary"):
        st.switch_page("Koemitsu.py") # Ganti jadi Koemitsu.py jika nama file utamamu sudah diganti
    st.stop() # Hentikan kodingan di bawahnya agar tidak error

# Pastikan variabel calibration_result ada (minimal isinya None)
if 'calibration_result' not in st.session_state:
    st.session_state.calibration_result = None

# ==========================================

render_audio_settings_sidebar()

hero_box('Langkah 1 — Kalibrasi', 'Cari nada ternyamanmu.') 

left, right = st.columns(2)
with left:
    card_open('Instruksi')
    render_calibration_guide()
    card_close()

with right:
    card_open('Rekam')
    calibration_file, _ = render_audio_capture(key_prefix='cal', label='Ambil suara')
    if st.button('Analisis', type='primary'):
        if calibration_file:
            with st.spinner("Menganalisis nada nyamanmu..."):
                _, y, suffix = load_audio_source(calibration_file)
                result = analyze_calibration_take(y)
                st.session_state.calibration_result = result
                st.session_state.selected_root_midi = result['suggestion'].root_midi
                st.session_state.steps = result['suggestion'].steps
                st.success("Berhasil!")
        else:
            st.error("Masukkan audio terlebih dahulu!")
    card_close()

# --- PERBAIKAN ERROR BARIS 30 ---
# Gunakan .get() agar aman jika variabelnya belum terisi
hasil_kalibrasi = st.session_state.get('calibration_result')

if hasil_kalibrasi:
    st.markdown("---")
    st.success(f"### 🎉 Nada Nyaman Terdeteksi: {hasil_kalibrasi['suggestion'].comfort_note_display}")
    st.write(f"Tangga nada pengujian kamu akan menggunakan dasar **{hasil_kalibrasi['suggestion'].key_name}**.")
    
    if st.button("Lanjut ke Tes Nada ➔", type="primary", use_container_width=True):
        st.switch_page("pages/2_Test.py")