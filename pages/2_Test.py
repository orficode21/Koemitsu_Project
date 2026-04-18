import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from src.inference import load_vocal_model
from src.analysis import analyze_note_take
from src.ui_helpers import (
    hero_box, 
    render_audio_capture,
    render_note_guide, 
    render_progress_pills
)
from src.audio_io import load_audio_source, save_audio_bytes
from src.storage import session_folder
from src.music_logic import ScaleStep, display_note # Import ScaleStep untuk fitur extended

# -- KONFIGURASI HALAMAN --
st.set_page_config(page_title='Tes Nada - Koemitsu', layout='wide')

# -- FUNGSI BANTUAN UNTUK MENAMPILKAN SPEKTROGRAM DI WEB --
def plot_mel_spectrogram(y, sr=22050):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(4, 2))
    img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='magma', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.set_title('Mel-Spectrogram Suaramu')
    plt.tight_layout()
    return fig

# -- 1. PROTEKSI & INISIALISASI DATA --
if 'steps' not in st.session_state or not st.session_state.steps:
    st.warning("⚠️ Silakan lakukan kalibrasi terlebih dahulu!")
    if st.button("Ke Halaman Kalibrasi"): st.switch_page("pages/1_Calibration.py")
    st.stop()

model = load_vocal_model()
steps = st.session_state.steps
num_steps = len(steps)

# Pastikan laci penyimpanan hasil sinkron dengan jumlah steps (antisipasi fitur extended)
if 'step_results' not in st.session_state:
    st.session_state.step_results = [None] * num_steps
elif len(st.session_state.step_results) < num_steps:
    # Jika jumlah steps bertambah (Extended), tambah slot di results
    diff = num_steps - len(st.session_state.step_results)
    st.session_state.step_results.extend([None] * diff)

if 'current_step_index' not in st.session_state:
    st.session_state.current_step_index = 0

idx = st.session_state.current_step_index
current_step = steps[idx]

# -- 2. HEADER & PROGRESS --
hero_box(
    f'Uji Nada: {current_step.degree}', 
    f"Target: {current_step.target_note_display} | Tangga Nada: {st.session_state.calibration_result['suggestion'].key_name}"
)
render_progress_pills(steps, completed=sum(1 for x in st.session_state.step_results if x is not None))

# -- 3. AREA REKAMAN (CARD) --
with st.container(border=True):
    col_info, col_action = st.columns([1, 1], gap="large")
    
    with col_info:
        st.markdown(f"### 🎵 Nada: **{current_step.degree}**")
        st.write(f"Dengarkan guide dan nyanyikan bunyi **'aaa'**.")
        render_note_guide(current_step.target_note, degree_slug=current_step.degree_slug)
        
        st.divider()

        # --- TAMPILAN HASIL (VERSI PRESENTASE & GRAFIK) ---
        if st.session_state.step_results[idx] is not None:
            res = st.session_state.step_results[idx]
            
            is_ok = res.get('pitch_ok', False)
            cents = res.get('cents_error', 0)
            score = res.get('ai_score', 0)
            # Menggunakan threshold dinamis dari session state
            threshold = float(st.session_state.get('res_threshold', 0.55))

            persen_resonansi = int(score * 100)
            toleransi = float(st.session_state.get('pitch_tolerance', 150))
            persen_pitch = max(0, 100 - int((abs(cents) / toleransi) * 100))

            if is_ok and score >= threshold:
                st.success("### 🌟 LUAR BIASA! NADA & KUALITAS PAS")
            elif is_ok and score < threshold:
                st.warning("### 👍 NADA PAS, TAPI SUARA TEGANG (STRAINED)")
            elif not is_ok:
                st.error("### 😅 OOPS! NADA BELUM PAS")

            c1, c2 = st.columns(2)
            c1.metric("Akurasi Nada (Pitch)", f"{persen_pitch}%", help=f"Meleset {cents} cents")
            
            if score >= threshold:
                c2.metric("Kualitas Vokal (AI)", f"{persen_resonansi}% Resonant", delta="Aman")
            else:
                c2.metric("Kualitas Vokal (AI)", f"{100 - persen_resonansi}% Strained", delta="Tegang", delta_color="inverse")
                
            if f"audio_y_{idx}" in st.session_state:
                y_audio = st.session_state[f"audio_y_{idx}"]
                st.pyplot(plot_mel_spectrogram(y_audio))

        else:
            st.info("🕒 Belum ada suara yang direkam. Yuk, coba tekan rekam di sebelah kanan!")

    with col_action:
        st.markdown("### 🎤 Rekam / Rekam Ulang")
        
        take_file, audio_bytes_raw = render_audio_capture(key_prefix=f"step_rec_{idx}", label=f"Mulai Rekam {current_step.degree}")
        
        if take_file: st.audio(take_file)
        
        if st.button("Simpan & Analisis Sekarang", type='primary', use_container_width=True):
            if take_file is None:
                st.error('Rekam suaramu dulu ya!')
            else:
                with st.spinner('Menganalisis suara & membuat spektrogram...'):
                    audio_raw, y, suffix = load_audio_source(take_file)
                    # Ambil gain dari session state
                    y = y * st.session_state.get('mic_gain', 1.0) 
                    
                    st.session_state[f"audio_y_{idx}"] = y 
                    
                    folder = session_folder(st.session_state.session_id)
                    save_audio_bytes(audio_raw, folder / f"step_{idx}_{current_step.degree_slug}{suffix}")
                    
                    row = analyze_note_take(
                        y=y, target_note=current_step.target_note, target_note_display=current_step.target_note_display,
                        degree=current_step.degree, model=model, res_threshold=float(st.session_state.res_threshold),
                        tolerance_cents=float(st.session_state.pitch_tolerance)
                    )
                    
                    st.session_state.step_results[idx] = row
                    st.rerun()

# -- 4. FITUR EXTENDED CHALLENGE (Hanya muncul di nada terakhir jika hasil bagus) --
current_done = st.session_state.step_results[idx] is not None
is_last_step = (idx == num_steps - 1)

if is_last_step and current_done:
    last_res = st.session_state.step_results[idx]
    # Syarat: Resonansi AI > 70% dan Pitch OK
    if last_res.get('ai_score', 0) >= 0.70 and last_res.get('pitch_ok'):
        st.markdown("---")
        st.balloons()
        st.info("🔥 **Wah, suaramu masih sangat stabil!** AI mendeteksi kamu masih bisa menyanyi lebih tinggi lagi. Ingin mencoba tantangan tambahan?")
        
        if st.button("🚀 YA! COBA 4 NADA LEBIH TINGGI", use_container_width=True):
            # Definisikan interval tambahan (Re, Mi, Fa, Sol di oktaf atas)
            # 14 = Re, 16 = Mi, 17 = Fa, 19 = Sol
            extra_intervals = [14, 16, 17, 19]
            extra_labels = ['Re (+)', 'Mi (+)', 'Fa (+)', 'Sol (+)']
            
            root_midi = st.session_state.calibration_result['suggestion'].root_midi
            
            for e_label, interval in zip(extra_labels, extra_intervals):
                new_midi = root_midi + interval
                new_note_name = librosa.midi_to_note(new_midi, octave=True)
                
                st.session_state.steps.append(ScaleStep(
                    index=len(st.session_state.steps) + 1,
                    degree=e_label,
                    degree_slug=e_label.lower().replace(' (+)', '_plus'),
                    target_midi=new_midi,
                    target_note=new_note_name,
                    target_note_display=display_note(new_note_name)
                ))
                # Tambah slot kosong di hasil
                st.session_state.step_results.append(None)
            
            # Pindah ke index nada baru yang baru saja ditambahkan
            st.session_state.current_step_index += 1
            st.rerun()

# -- 5. TOMBOL NAVIGASI --
st.divider()
nav_prev, nav_status, nav_next = st.columns([1, 2, 1])

with nav_prev:
    if idx > 0:
        if st.button("⬅ Sebelumnya", use_container_width=True):
            st.session_state.current_step_index -= 1
            st.rerun()

with nav_status:
    st.markdown(f"<p style='text-align: center; padding-top: 10px;'>Nada {idx+1} dari {len(st.session_state.steps)}</p>", unsafe_allow_html=True)

with nav_next:
    # Tombol "Lihat Hasil" hanya muncul jika berada di nada PALING AKHIR (termasuk nada tambahan)
    if idx == len(st.session_state.steps) - 1:
        if st.button("🏁 LIHAT HASIL", type='primary', use_container_width=True, disabled=not current_done):
            st.switch_page("pages/3_Result.py")
    else:
        if st.button("Berikutnya ➡", use_container_width=True, disabled=not current_done):
            st.session_state.current_step_index += 1
            st.rerun()