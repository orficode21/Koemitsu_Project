import numpy as np
from scipy.io import wavfile

def generate_rich_vocal_guide(filename, start_note_midi, num_notes=9, duration_per_note=1.5):
    sr = 22050
    # Interval Tangga Nada Mayor: 1.5 Oktaf
    major_intervals = [0, 2, 4, 5, 7, 9, 11, 12, 14] 
    
    full_audio = np.array([])

    for interval in major_intervals:
        midi_note = start_note_midi + interval
        freq = 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))
        
        t = np.linspace(0, duration_per_note, int(sr * duration_per_note), endpoint=False)
        
        # --- MEMBUAT SUARA LEBIH KAYA (COMPOSITE WAVE) ---
        # Nada dasar (Fundamental)
        wave = 0.5 * np.sin(2 * np.pi * freq * t)
        # Tambah Harmonik ke-1 (agar lebih 'cerah')
        wave += 0.25 * np.sin(2 * np.pi * (2 * freq) * t)
        # Tambah Harmonik ke-2 (agar lebih 'tegas')
        wave += 0.1 * np.sin(2 * np.pi * (3 * freq) * t)
        
        # Tambahkan ADSR Envelope sederhana agar suara tidak mendadak mati (biar halus)
        fade_in = np.linspace(0, 1, int(sr * 0.1))
        fade_out = np.linspace(1, 0, int(sr * 0.2))
        envelope = np.ones_like(wave)
        envelope[:len(fade_in)] = fade_in
        envelope[-len(fade_out):] = fade_out
        
        wave *= envelope
        full_audio = np.concatenate([full_audio, wave])

    # Normalisasi volume agar maksimal (tidak pecah tapi keras)
    full_audio = full_audio / np.max(np.abs(full_audio))
    full_audio = (full_audio * 32767).astype(np.int16)
    
    wavfile.write(filename, sr, full_audio)
    print(f"File {filename} (Rich Sound) berhasil dibuat!")

# --- EKSEKUSI ---

# Laki-laki: Start C3 (MIDI 48) - Ini standar Baritone
generate_rich_vocal_guide("guide_male.wav", start_note_midi=48)

# Perempuan: Start C4 (MIDI 60) - Ini standar Soprano/Mezzo
generate_rich_vocal_guide("guide_female.wav", start_note_midi=60)