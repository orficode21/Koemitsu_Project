import os
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm

# =========================
# KONFIGURASI
# =========================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODEL_PATH = "vocal_model_loso.h5"
SR = 22050

# Durasi nada saat scale test (UX paling gampang)
NOTE_DURATION = 1.0          # 1 detik per nada
NOTE_HOP = 1.0               # geser 1 detik (tanpa overlap antar nada)
SAMPLES_PER_NOTE = int(SR * NOTE_DURATION)

# Voting internal dalam 1 nada (biar lebih stabil)
SLICE_WIN = 0.5              # 0.5 detik
SLICE_HOP = 0.25             # 0.25 detik
SAMPLES_SLICE_WIN = int(SR * SLICE_WIN)
SAMPLES_SLICE_HOP = int(SR * SLICE_HOP)

# Pitch detection
PYIN_FMIN = librosa.note_to_hz("C2")
PYIN_FMAX = librosa.note_to_hz("C6")

# Threshold resonant
RES_THRESHOLD = 0.55         # bisa kamu tune (0.5-0.65)

# =========================
# Contoh lagu offline (TA-friendly)
# Key di sini pakai pitch class: C, C#, D, Eb, ...
# =========================
SONG_DB = {
    "C": ["Let It Be - The Beatles", "Someone Like You - Adele (umum dimainkan di C)", "Perfect - Ed Sheeran (sering di C)"],
    "G": ["Perfect - Ed Sheeran (sering di G)", "Yellow - Coldplay (sering di G)", "Stand By Me - Ben E. King (sering di G)"],
    "D": ["I'm Yours - Jason Mraz (sering di D)", "Wonderwall - Oasis (sering di D/Em tergantung versi)"],
    "A": ["I'm Yours - Jason Mraz (sering di A versi capo)", "Hallelujah (variasi, banyak versi di A)"],
    "E": ["Hotel California (sering dimainkan di Em, tapi referensi E/E minor umum)", "Thinking Out Loud (banyak versi E)"],
    "F": ["All of Me - John Legend (sering di F)", "Someone Like You (sering di F versi piano)"],
    "Bb": ["All of Me (banyak versi Bb)", "Just The Way You Are (banyak versi Bb)"],
    "Eb": ["Rolling in the Deep (banyak versi Eb)"],
    "Ab": ["Someone Like You (sering Ab versi tertentu)"],
    "B": ["Thinking Out Loud (versi tertentu B)"],
    "C#": ["(contoh) Lagu pop sering ditranspose ke C# untuk vokalis tertentu"],
    "F#": ["(contoh) Lagu worship sering di F#"],
}

# =========================
# UTIL MUSIK
# =========================
NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC_FLAT_MAP = {"A#": "Bb", "D#": "Eb", "G#": "Ab"}

def pitch_class(note_with_octave: str) -> str:
    """C4 -> C, D#3 -> D#, Bb3 -> A# (librosa biasanya output sharps)"""
    s = "".join([c for c in note_with_octave if not c.isdigit()])
    return s

def to_flat_if_common(pc: str) -> str:
    """Ubah A# -> Bb, dst, biar user-friendly."""
    return ENHARMONIC_FLAT_MAP.get(pc, pc)

def midi_to_pitch_class(m: int) -> str:
    pc = NOTE_NAMES_SHARP[m % 12]
    return to_flat_if_common(pc)

def snap_do_to_nearest_C_below(median_midi: int) -> int:
    """Pilih Do sebagai C terdekat di bawah nada median (stabil & gampang dijelaskan)."""
    # Pitch class C = 0
    pc = median_midi % 12
    diff_down = pc  # berapa semitone turun untuk sampai C
    do_midi = median_midi - diff_down
    # kalau do_midi sama dengan median (berarti median pitch class C), turunin satu oktaf biar aman
    if do_midi == median_midi:
        do_midi -= 12
    return do_midi

def build_major_scale_one_octave(do_midi: int):
    """Major scale intervals: 0,2,4,5,7,9,11,12"""
    intervals = [0, 2, 4, 5, 7, 9, 11, 12]
    notes = []
    for i in intervals:
        m = do_midi + i
        note = librosa.midi_to_note(m)
        notes.append(note)
    return notes

# =========================
# MODEL LOADER
# =========================
@st.cache_resource
def load_vocal_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model .h5: {e}")
        return None

# =========================
# AUDIO HELPERS
# =========================
def load_audio_from_upload(uploaded_file):
    y, _ = librosa.load(uploaded_file, sr=SR)
    return y

def load_audio_from_bytes(audio_bytes):
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR)
    return y

def safe_median_f0(f0):
    f0 = np.array(f0)
    f0 = f0[~np.isnan(f0)]
    f0 = f0[f0 > 0]
    if len(f0) == 0:
        return None
    return float(np.median(f0))

def detect_comfort_note(y):
    """Deteksi nada nyaman (median F0) dari rekaman kalibrasi."""
    f0, _, _ = librosa.pyin(y, fmin=PYIN_FMIN, fmax=PYIN_FMAX, sr=SR)
    med = safe_median_f0(f0)
    if med is None:
        return None, None
    note = librosa.hz_to_note(med)
    return med, note

def preprocess_chunk_to_rgb_magma(chunk):
    """Buat input CNN sesuai training: mel_db -> normalize -> magma RGB -> resize 128x128."""
    mel = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # normalize -80..0 dB -> 0..1
    mel_norm = np.clip((mel_db + 80) / 80, 0, 1)
    colored = cm.magma(mel_norm)[..., :3]  # drop alpha

    x = tf.image.resize(colored, [128, 128]).numpy()
    x = np.expand_dims(x, axis=0)  # batch
    return x

def score_resonance_for_note_segment(y_note, model):
    """
    Voting: potong y_note (durasi 1 detik) jadi beberapa slice kecil,
    prediksi per slice, ambil median.
    """
    scores = []
    # pastikan panjang minimal
    if len(y_note) < SAMPLES_SLICE_WIN:
        # pad
        y_note = np.pad(y_note, (0, SAMPLES_SLICE_WIN - len(y_note)), mode="constant")

    for start in range(0, max(1, len(y_note) - SAMPLES_SLICE_WIN + 1), SAMPLES_SLICE_HOP):
        chunk = y_note[start:start + SAMPLES_SLICE_WIN]
        x = preprocess_chunk_to_rgb_magma(chunk)
        pred = model.predict(x, verbose=0)
        scores.append(float(pred[0][0]))

    if len(scores) == 0:
        return None
    return float(np.median(scores))

def detect_note_in_segment(y_note):
    """Deteksi pitch (median) pada segmen 1 nada."""
    f0, _, _ = librosa.pyin(y_note, fmin=PYIN_FMIN, fmax=PYIN_FMAX, sr=SR)
    med = safe_median_f0(f0)
    if med is None:
        return None, "N/A"
    note = librosa.hz_to_note(med)
    return med, note

def cents_distance(note_a, note_b):
    """Jarak dalam cents antara dua note (berdasarkan MIDI)."""
    try:
        ma = librosa.note_to_midi(note_a)
        mb = librosa.note_to_midi(note_b)
    except Exception:
        return None
    return abs(ma - mb) * 100

# =========================
# CORE ANALYSIS: SCALE TEST
# =========================
def analyze_scale_test(y, model, target_notes):
    """
    y: audio user nyanyi scale (durasi kira-kira len(target_notes)*NOTE_DURATION)
    target_notes: list seperti ["C3","D3",...,"C4"]
    """
    results = []
    total_notes = len(target_notes)

    prog = st.progress(0)
    for i, target in enumerate(target_notes):
        start = int(i * NOTE_HOP * SR)
        end = start + SAMPLES_PER_NOTE
        if start >= len(y):
            break

        seg = y[start:end]
        if len(seg) < SAMPLES_PER_NOTE:
            seg = np.pad(seg, (0, SAMPLES_PER_NOTE - len(seg)), mode="constant")

        # pitch detection per segmen
        f0_med, note_detected = detect_note_in_segment(seg)

        # CNN resonance score (median voting)
        score = score_resonance_for_note_segment(seg, model)

        # validasi pitch (opsional)
        pitch_ok = False
        cents = None
        if note_detected != "N/A":
            cents = cents_distance(note_detected, target)
            if cents is not None and cents <= 150:  # toleransi 150 cents (1.5 semitone)
                pitch_ok = True

        if score is None:
            quality = "N/A"
        else:
            quality = "Resonant" if score >= RES_THRESHOLD else "Strained"

        results.append({
            "Index": i+1,
            "Target Nada": target,
            "Nada Terdeteksi": note_detected,
            "Pitch OK": pitch_ok,
            "Cents Error": None if cents is None else round(cents, 1),
            "Skor AI": None if score is None else round(score, 4),
            "Kualitas": quality
        })

        prog.progress(min(1.0, (i+1)/max(1, total_notes)))

    return pd.DataFrame(results)

# =========================
# REKOMENDASI MUSIK (CEILING + KEY)
# =========================
def get_musical_recommendation_from_scale(df):
    """
    df berisi per nada (target+detected+score).
    Ceiling = nada target tertinggi yang (Pitch OK) dan Resonant konsisten.
    """
    dfv = df.copy()

    # Ambil yang valid (pitch OK) dan skor tersedia
    dfv = dfv[(dfv["Skor AI"].notna()) & (dfv["Pitch OK"] == True)].copy()
    if dfv.empty:
        return None, None, None, None

    # Agregasi per Target Nada (karena setiap nada 1 baris, ini sederhana)
    dfv["target_midi"] = dfv["Target Nada"].apply(librosa.note_to_midi)

    # Kandidat resonant
    df_res = dfv[dfv["Skor AI"] >= RES_THRESHOLD].copy()
    if df_res.empty:
        return None, None, None, None

    ceiling_midi = int(df_res["target_midi"].max())
    ceiling_note = librosa.midi_to_note(ceiling_midi)

    # Root = ceiling - 7 semitone (Perfect Fifth down)
    root_midi = ceiling_midi - 7
    root_note = librosa.midi_to_note(root_midi)
    key_pc = midi_to_pitch_class(root_midi)
    key_name = f"{key_pc} Mayor"

    return ceiling_note, key_name, root_note, key_pc

# =========================
# UI
# =========================
def main():
    st.set_page_config(page_title="Vocal Tessitura AI", page_icon="🎤", layout="wide")
    st.title("🎤 Vocal Tessitura AI & Key Recommender")
    st.caption("Flow: Kalibrasi nada nyaman → Tentukan Do awal otomatis → Nyanyi scale → Deteksi Ceiling → Rekomendasi Key.")

    model = load_vocal_model()
    if model is None:
        st.stop()

    # Sidebar
    st.sidebar.header("⚙️ Mode & Pengaturan")

    mode = st.sidebar.radio("Mode Uji", [
        "1) Kalibrasi + Scale Otomatis (Recommended)",
        "2) Langsung Scale (Manual Do)"
    ])

    st.sidebar.markdown("---")
    st.sidebar.write("Threshold Resonant")
    thr = st.sidebar.slider("RES_THRESHOLD", 0.40, 0.80, float(RES_THRESHOLD), 0.01)
    global RES_THRESHOLD
    RES_THRESHOLD = thr

    st.sidebar.markdown("---")
    st.sidebar.write("Durasi per nada (detik)")
    dur = st.sidebar.selectbox("NOTE_DURATION", [0.75, 1.0, 1.25], index=1)
    global NOTE_DURATION, NOTE_HOP, SAMPLES_PER_NOTE
    NOTE_DURATION = float(dur)
    NOTE_HOP = float(dur)
    SAMPLES_PER_NOTE = int(SR * NOTE_DURATION)

    st.sidebar.markdown("---")
    st.sidebar.write("🎵 Petunjuk")
    st.sidebar.caption("1) Nyanyi 'AAAA' stabil.\n2) Ikuti guide tempo: 1 nada = 1 durasi.\n3) Stop kalau sudah capek.")

    # Session state
    if "comfort_note" not in st.session_state:
        st.session_state.comfort_note = None
    if "do_midi" not in st.session_state:
        st.session_state.do_midi = None
    if "target_scale" not in st.session_state:
        st.session_state.target_scale = None

    # =========================
    # INPUT AUDIO
    # =========================
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎙️ Rekam / Upload untuk Kalibrasi (Nada Nyaman)")
        st.caption("Nyanyikan 2–3 detik nada bebas paling nyaman pakai 'AAAA'.")

        calib_audio = None
        up_calib = st.file_uploader("Upload kalibrasi (WAV/MP3)", type=["wav", "mp3"], key="up_calib")
        if up_calib:
            st.audio(up_calib)
            calib_audio = load_audio_from_upload(up_calib)

        try:
            from audio_recorder_streamlit import audio_recorder
            calib_bytes = audio_recorder(text="Rekam Kalibrasi", pause_threshold=1.5, key="rec_calib")
            if calib_bytes:
                st.audio(calib_bytes, format="audio/wav")
                calib_audio = load_audio_from_bytes(calib_bytes)
        except:
            st.info("Jika ingin rekam langsung, install: audio-recorder-streamlit")

        if st.button("🎯 Jalankan Kalibrasi", disabled=(mode.startswith("2)") or calib_audio is None)):
            hz, note = detect_comfort_note(calib_audio)
            if note is None:
                st.error("Pitch tidak terdeteksi. Coba nyanyi lebih jelas/lebih stabil.")
            else:
                st.session_state.comfort_note = note
                med_midi = int(librosa.note_to_midi(note))
                do_midi = snap_do_to_nearest_C_below(med_midi)
                st.session_state.do_midi = do_midi
                st.session_state.target_scale = build_major_scale_one_octave(do_midi)
                st.success(f"✅ Nada nyaman terdeteksi: **{note}** (~{hz:.1f} Hz)")
                st.info(f"➡️ Do awal otomatis dipilih: **{librosa.midi_to_note(do_midi)}**")
                st.write("Target scale:", " → ".join(st.session_state.target_scale))

    with col2:
        st.subheader("🎶 Rekam / Upload untuk Scale Test")
        st.caption("Nyanyikan scale 'AAAA' sesuai target (1 nada per durasi).")

        # Manual Do jika mode 2
        if mode.startswith("2)"):
            manual_do = st.selectbox("Pilih Do awal (manual)", ["C3", "C4", "D3", "D4", "E3", "F3", "G3", "A3"], index=0)
            st.session_state.do_midi = int(librosa.note_to_midi(manual_do))
            st.session_state.target_scale = build_major_scale_one_octave(st.session_state.do_midi)
            st.write("Target scale:", " → ".join(st.session_state.target_scale))
        else:
            if st.session_state.target_scale is None:
                st.warning("Lakukan kalibrasi dulu agar Do awal otomatis ter-set.")
            else:
                st.write("Target scale:", " → ".join(st.session_state.target_scale))

        scale_audio = None
        up_scale = st.file_uploader("Upload scale test (WAV/MP3)", type=["wav", "mp3"], key="up_scale")
        if up_scale:
            st.audio(up_scale)
            scale_audio = load_audio_from_upload(up_scale)

        try:
            from audio_recorder_streamlit import audio_recorder
            scale_bytes = audio_recorder(text="Rekam Scale Test", pause_threshold=2.0, key="rec_scale")
            if scale_bytes:
                st.audio(scale_bytes, format="audio/wav")
                scale_audio = load_audio_from_bytes(scale_bytes)
        except:
            pass

        can_run = (scale_audio is not None) and (st.session_state.target_scale is not None)

        if st.button("🚀 MULAI ANALISIS SCALE", type="primary", disabled=not can_run):
            with st.spinner("Menganalisis pitch + kualitas resonansi per nada..."):
                df = analyze_scale_test(scale_audio, model, st.session_state.target_scale)

                ceiling, key_name, root_note, key_pc = get_musical_recommendation_from_scale(df)

                st.divider()
                st.subheader("📋 Hasil Analisis Tessitura")

                c1, c2, c3 = st.columns(3)
                with c1:
                    if ceiling is None:
                        st.metric("Vocal Ceiling", "Tidak terdeteksi")
                    else:
                        st.metric("Vocal Ceiling (Atap Aman)", ceiling)
                    st.caption("Nada tertinggi (target) yang masih resonant & pitch valid.")

                with c2:
                    if key_name is None:
                        st.error("Gagal memberi rekomendasi key. Coba nyanyi lebih stabil.")
                    else:
                        st.success(f"### Rekomendasi Nada Dasar: {key_name}")
                        st.caption(f"Logika: Root = Ceiling − 7 semitone (Perfect Fifth).")

                with c3:
                    valid_rows = df[df["Pitch OK"] == True]
                    if len(valid_rows) > 0 and df["Skor AI"].notna().any():
                        resonant_rate = (valid_rows["Kualitas"] == "Resonant").mean() * 100
                        st.metric("Resonant Rate (Pitch OK)", f"{resonant_rate:.1f}%")
                    else:
                        st.metric("Resonant Rate", "N/A")

                # Grafik skor per nada (target)
                st.divider()
                st.subheader("📊 Skor AI per Nada (Target)")

                df_plot = df.copy()
                df_plot["Skor AI"] = pd.to_numeric(df_plot["Skor AI"], errors="coerce")
                df_plot = df_plot[df_plot["Skor AI"].notna()].copy()

                if not df_plot.empty:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    colors = []
                    for _, row in df_plot.iterrows():
                        if row["Pitch OK"] != True:
                            colors.append("#95a5a6")  # gray for invalid pitch
                        else:
                            colors.append("#2ecc71" if row["Kualitas"] == "Resonant" else "#e74c3c")

                    ax.bar(df_plot["Target Nada"], df_plot["Skor AI"], color=colors)
                    ax.axhline(RES_THRESHOLD, color="black", linestyle="--", alpha=0.6, label="Threshold")

                    if ceiling is not None:
                        ax.set_title(f"Ceiling: {ceiling} | Recommended Key: {key_name}")
                        # tandai ceiling
                        ax.annotate(
                            f"CEILING {ceiling}",
                            xy=(ceiling, RES_THRESHOLD),
                            xytext=(ceiling, 0.95),
                            arrowprops=dict(arrowstyle="->"),
                            fontsize=10,
                            fontweight="bold"
                        )

                    ax.set_ylabel("Skor Resonansi (AI)")
                    ax.set_ylim(0, 1.05)
                    plt.xticks(rotation=45)
                    ax.legend()
                    st.pyplot(fig)

                # Tabel detail
                with st.expander("🔍 Detail per Nada"):
                    st.dataframe(df, use_container_width=True)

                # Contoh lagu
                st.divider()
                st.subheader("🎵 Contoh Lagu Populer (Offline)")

                if key_pc is None:
                    st.info("Belum ada key recommendation yang valid.")
                else:
                    songs = SONG_DB.get(key_pc, [])
                    if len(songs) == 0:
                        st.warning(f"Belum ada database lagu untuk key {key_pc}. Kamu bisa tambah di SONG_DB.")
                    else:
                        st.write(f"Berikut contoh lagu yang sering dikenal (atau sering dimainkan) di **{key_pc}**:")
                        for s in songs[:5]:
                            st.write(f"- {s}")

    st.caption("Tips: kalau pitch sering 'N/A', coba rekam lebih dekat mic dan nyanyi lebih stabil (vokal 'AAAA').")

if __name__ == "__main__":
    main()