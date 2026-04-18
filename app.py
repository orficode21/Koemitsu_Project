import streamlit as st
from src.ui_helpers import inject_css, hero_box, card_open, card_close
from src.inference import load_vocal_model, find_model_path
from src.storage import ensure_dirs, new_session_id

# Konfigurasi Halaman
st.set_page_config(page_title='Koemitsu', page_icon='🎵', layout='wide')
inject_css()
ensure_dirs()

# Inisialisasi State (Agar data tersimpan saat pindah halaman)
def init_state():
    if 'session_id' not in st.session_state:
        st.session_state.update({
            'session_id': new_session_id(),
            'res_threshold': 0.6,
            'pitch_tolerance': 100,
            'calibration_result': None,
            'selected_root_midi': None,
            'steps': [],
            'current_step_index': 0,
            'step_results': [],
            'saved_calibration': False,
            'saved_summary': False,
        })

init_state()

# Sidebar (Muncul di semua halaman)
with st.sidebar:
    st.title('Koemitsu')
    st.write(f"**ID Sesi**: `{st.session_state.session_id}`")
    if st.button('Mulai Sesi Baru'):
        st.session_state.clear()
        st.rerun()

# Tampilan Intro
hero_box('Koemitsu — Temukan Suaramu!', 'Temukan Nada Dasarmu.')

col1, col2 = st.columns([1, 1])
with col1:
    card_open('Alur Proyek')
    st.markdown("1. Kalibrasi\n2. Tes Nada\n3. Hasil")
    card_close()

with col2:
    card_open('Mulai')
    st.write("Siap untuk memulai analisis?")
    if st.button('Pergi ke Kalibrasi', type='primary'):
        st.switch_page("pages/1_Calibration.py")
    card_close()