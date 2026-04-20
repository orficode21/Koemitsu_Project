from __future__ import annotations

from typing import Optional
import sounddevice as sd
import streamlit as st

from src.audio_io import allowed_audio_help_text, find_calibration_guide, get_guide_audio_bytes, get_uploaded_bytes

def render_device_settings():
    """
    Menampilkan pengaturan Input/Output device di dalam halaman.
    """
    with st.expander("🔊 Pengaturan Perangkat & Audio (Hanya Lokal)", expanded=False):
        st.info("Pilih perangkat yang ingin Anda gunakan. Pastikan mic sudah tercolok.")
        
        # 1. Ambil daftar hardware dari laptop secara real-time
        devices = sd.query_devices()
        input_devices = [f"{d['name']} (Input)" for d in devices if d['max_input_channels'] > 0]
        output_devices = [f"{d['name']} (Output)" for d in devices if d['max_output_channels'] > 0]

        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox(
                "Pilih Microphone (Input)", 
                input_devices, 
                key="selected_input_device"
            )
            st.session_state.mic_gain = st.slider(
                "Gain (Volume Input)", 0.5, 3.0, 1.0, 0.1, key="gain_slider"
            )

        with col2:
            st.selectbox(
                "Pilih Speaker (Output)", 
                output_devices, 
                key="selected_output_device"
            )
            st.session_state.noise_threshold = st.slider(
                "Noise Threshold", 0.00, 0.05, 0.01, 0.005, key="noise_slider"
            )

        st.warning("⚠️ **Catatan Penting:** Pemilihan perangkat di atas hanya untuk referensi sistem. Untuk perekaman di Browser, Anda **WAJIB** tetap memilih mic yang sama melalui ikon 'Kamera/Mic' di sebelah link URL (localhost).")

def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1050px;}
        .hero {
            background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(16,185,129,0.10));
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 22px;
            padding: 1.4rem 1.4rem 1.1rem 1.4rem;
            margin-bottom: 1rem;
        }
        .card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 18px;
            padding: 1rem 1rem 0.8rem 1rem;
            margin-bottom: 0.9rem;
        }
        .step-pill {
            display: inline-block;
            padding: 0.30rem 0.65rem;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.25);
            margin: 0.1rem 0.2rem 0.2rem 0;
            font-size: 0.92rem;
        }
        .metric-chip {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            background: rgba(15,23,42,0.08);
            border: 1px solid rgba(148,163,184,0.18);
            margin-right: 0.35rem;
            margin-bottom: 0.3rem;
            font-size: 0.9rem;
        }
        .small-muted {color: #94a3b8; font-size: 0.93rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

def hero_box(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class='hero'>
            <h2 style='margin:0 0 0.35rem 0'>{title}</h2>
            <div class='small-muted'>{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_open(title: Optional[str] = None):
    if title:
        st.markdown(f"<div class='card'><h4 style='margin-top:0'>{title}</h4>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)


def card_close():
    st.markdown('</div>', unsafe_allow_html=True)


def render_audio_capture(key_prefix: str, label: str, help_text: Optional[str] = None):
    st.write(label)
    
    # Hanya menampilkan input rekaman suara (Real-time)
    audio_file = None
    if hasattr(st, 'audio_input'):
        audio_file = st.audio_input('Klik tombol di bawah untuk mulai merekam', key=f'{key_prefix}_mic', help=help_text)
    
    # Variabel chosen sekarang hanya mengambil dari audio_file (perekam)
    chosen = audio_file
    
    audio_bytes = get_uploaded_bytes(chosen)
    
    # Menampilkan pemutar suara jika rekaman sudah ada (untuk cek ulang)
    if audio_bytes:
        st.audio(audio_bytes)
        
    return chosen, audio_bytes


def render_calibration_guide() -> None:
    guide = find_calibration_guide()
    if guide and guide.exists():
        st.caption('Contoh vokal kalibrasi')
        st.audio(guide.read_bytes(), format='audio/wav')
    else:
        # INI YANG KAMU UBAH:
        st.markdown("""
        **Cara Kalibrasi:**
        1. Tarik napas dalam dan rileks.
        2. Klik/Tekan ikon mic untuk mulai merekam.
        3. Lantunkan bunyi **"aaa"** secara stabil selama **2-3 detik**.
        4. Gunakan nada yang menurutmu **paling enak dan nyaman** (Random aja).
        """)


def render_note_guide(target_note: str, degree_slug: str) -> None:
    audio_bytes, source = get_guide_audio_bytes(target_note, degree_slug=degree_slug)
    if source == 'custom':
        st.caption('Guide nada')
    else:
        st.caption('Guide nada (tone generator)')
    st.audio(audio_bytes, format='audio/wav')


def render_progress_pills(steps, completed: int) -> None:
    pills = []
    for idx, step in enumerate(steps, start=1):
        prefix = '✅' if idx <= completed else ('🎯' if idx == completed + 1 else '•')
        pills.append(f"<span class='step-pill'>{prefix} {step.degree}</span>")
    st.markdown(''.join(pills), unsafe_allow_html=True)

def render_audio_settings_sidebar():
    with st.sidebar:
        st.header("⚙️ Live Tuning AI")
        st.markdown("Gunakan ini untuk menyesuaikan sensitivitas AI dengan kondisi ruangan.")

        # 1. Slider Sensitivitas Resonansi (Ambang Batas)
        # Kita simpan langsung ke session_state
        st.session_state.res_threshold = st.slider(
            "Ambang Resonant (Threshold)", 
            min_value=0.10, 
            max_value=0.90, 
            value=st.session_state.get('res_threshold', 0.40), 
            step=0.05,
            help="Semakin rendah, AI semakin mudah memberikan label 'Resonant'."
        )

        # 2. Slider Gain (Volume Mic)
        st.session_state.mic_gain = st.slider(
            "Sensitivitas Mic (Gain)", 
            min_value=0.5, 
            max_value=3.0, 
            value=st.session_state.get('mic_gain', 1.0), 
            step=0.1,
            help="Naikkan jika suara responden terlalu pelan."
        )

        # 3. Slider Toleransi Pitch (Fals)
        st.session_state.pitch_tolerance = st.slider(
            "Toleransi Pitch (Cents)", 
            min_value=50, 
            max_value=300, 
            value=st.session_state.get('pitch_tolerance', 150), 
            step=10
        )

        st.markdown("---")
        if st.button("♻️ Reset ke Default"):
            st.session_state.res_threshold = 0.40
            st.session_state.mic_gain = 1.0
            st.session_state.pitch_tolerance = 150
            st.rerun()