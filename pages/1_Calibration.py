import streamlit as st
from src.audio_io import load_audio_source, save_audio_bytes
from src.analysis import analyze_calibration_take
from src.ui_helpers import hero_box, card_open, card_close, render_audio_capture
from src.music_logic import manual_root_options, display_note_from_midi, key_name_from_root_midi, build_major_scale_steps

st.set_page_config(page_title='Kalibrasi - Koemitsu', layout='wide')

hero_box('Langkah 1 — Kalibrasi', 'Lantunkan "aaa" selama 2 detik dengan nada yang menurutmu nyaman (tinggi)')

left, right = st.columns(2)
with left:
    card_open('Instruksi')
    card_close()

with right:
    card_open('Rekam')
    calibration_file, _ = render_audio_capture(key_prefix='cal', label='Ambil suara')
    if st.button('Analisis', type='primary'):
        if calibration_file:
            _, y, suffix = load_audio_source(calibration_file)
            result = analyze_calibration_take(y)
            st.session_state.calibration_result = result
            st.session_state.selected_root_midi = result['suggestion'].root_midi
            st.session_state.steps = result['suggestion'].steps
            st.success("Berhasil!")
    card_close()

# Tampilkan hasil jika sudah ada
if st.session_state.calibration_result:
    st.write(f"Nada Nyaman: {st.session_state.calibration_result['suggestion'].comfort_note_display}")
    if st.button("Lanjut ke Tes Nada"):
        st.switch_page("pages/2_Test.py")