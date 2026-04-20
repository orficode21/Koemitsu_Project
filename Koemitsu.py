import streamlit as st
from src.ui_helpers import inject_css, hero_box, card_open, card_close
from src.storage import ensure_dirs, new_session_id

# 1. Konfigurasi Halaman Utama
st.set_page_config(
    page_title='Koemitsu - Temukan Suaramu', 
    page_icon='🎵', 
    layout='wide',
    initial_sidebar_state="expanded"
)

inject_css()
ensure_dirs()

# 2. Inisialisasi State (RAM Aplikasi)
def init_state():
    if 'session_id' not in st.session_state:
        st.session_state.update({
            'session_id': new_session_id(),
            'user_name': '',            # Tetap simpan di background, input nanti di Result
            'music_pref': 'Pop Barat 🌍', 
            'res_threshold': 0.55,
            'pitch_tolerance': 150,
            'calibration_result': None,
            'selected_root_midi': None,
            'steps': [],
            'current_step_index': 0,
            'step_results': [],
            'saved_calibration': False,
            'saved_summary': False,
        })

init_state()

# 3. Sidebar (Branding Projek)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/461/461238.png", width=80) 
    st.title('Koemitsu')
    st.caption("Vocal Tessitura Classification & Key Recommendation")
    st.markdown("---")
    st.write(f"**ID Sesi**: `{st.session_state.session_id}`")
    
    if st.button('🔄 Reset Sesi', use_container_width=True):
        st.session_state.clear()
        st.rerun()

# 4. Tampilan Utama: Penjelasan Projek
hero_box(
    'Koemitsu: Temukan Suaramu!🎵', 
    'Aplikasi untuk membantumenemukan nada dasar lagu yang paling aman dan nyaman.'
)

st.markdown("""
### 💡 Tentang Projek Ini
Koemitsu adalah sistem cerdas yang menggabungkan **Digital Signal Processing (pYIN)** dan **Deep Learning (CNN)** untuk mengidentifikasi wilayah kenyamanan vokal manusia (*Tessitura*). 
Berbeda dengan alat pendeteksi nada biasa, Koemitsu tidak hanya melihat "sampai tidaknya" nada, tetapi menganalisis "sehat tidaknya" cara nada tersebut diproduksi.
""")

st.markdown("---")

# 5. Visualisasi Alur Pengujian
st.subheader("🛠️ Alur Pengujian")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 1. Kalibrasi Dasar")
    st.info("""
    **Mencari Nada untuk Do**
    \nSistem merekam nada yang menurut kamu paling nyaman (rileks). Ini digunakan sebagai titik awal (Lantai) untuk tangga nada pengujian.
    """)

with col2:
    st.markdown("#### 2. Uji Tangga Nada")
    st.warning("""
    **Analisis Kualitas Real-Time**
    \nKamu akan mengikuti panduan nada naik. AI (CNN) membedakan secara visual pada spektrogram apakah suaramu masih **Resonant** atau sudah **Strained**.
    """)

with col3:
    st.markdown("#### 3. Rekomendasi Cerdas")
    st.success("""
    **Penentuan Nada Dasar**
    \nSistem menghitung 'Atap' aman suaramu dan memberikan rekomendasi **Kunci Lagu (Key)** serta angka **Transpose** yang paling pas untukmu.
    """)

st.markdown("---")

# 6. Bagian Persiapan & Tombol Mulai
left_col, right_col = st.columns([1.2, 0.8], gap="large")

with left_col:
    card_open("📢 Persiapan Sebelum Memulai")
    st.markdown("""
    - **Gunakan Earphone:** Sangat disarankan agar suara piano *guide* tidak masuk kembali ke rekaman mikrofon.
    - **Cari Ruangan Tenang:** Hindari desis AC yang terlalu keras atau suara bising di latar belakang.
    - **Vokal 'Aaa':** Selama pengujian, gunakan vokal terbuka 'Aaa' agar AI dapat membedakan resonansi dengan akurat.
    """)
    card_close()

with right_col:
    card_open("🎙️ Siap Temukan Batas Suaramu?")
    st.write("Klik tombol di bawah untuk langsung menuju tahap kalibrasi nada nyaman.")
    
    if st.button('Mulai Analisis Sekarang ➔', type='primary', use_container_width=True):
        st.switch_page("pages/1_Calibration.py")
    card_close()

# 7. Footer Ilmiah
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🔍 Detail Teknis untuk Sidang/Exhibition"):
    st.write("""
    - **Model:** Convolutional Neural Network (CNN) 2D.
    - **Input:** Mel-Spectrogram (128x128 pixel).
    - **Logika Musik:** *Circle of Fifths* dengan asumsi klimaks melodi pada interval *Perfect Fifth* (7 semitone).
    - **Validasi:** Leave-One-Singer-Out Cross-Validation (LOSO-CV) dengan rata-rata akurasi 83.54%.
    """)

st.caption("Dikembangkan oleh Lutfi Mawardi - Tugas Akhir AI & Robotics  2026")