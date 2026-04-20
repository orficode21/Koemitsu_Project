import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
from PIL import Image

def preprocess_chunk_to_rgb_magma(chunk, sr=22050, image_size=(128, 128)):
    # ====================================================
    # 1. NOISE REDUCTION & AUDIO NORMALIZATION
    # ====================================================
    
    # A. Peak Normalization (Penguatan Suara Vokal)
    # Memastikan bagian paling keras (vokal) selalu berada di amplitudo 1.0
    max_amp = np.max(np.abs(chunk))
    if max_amp > 1e-6:
        chunk = chunk / max_amp
        
    # B. Noise Gate Sederhana (Penghapus Desis/Kipas)
    # Mematikan suara yang amplitudonya di bawah 2% (0.02) dari suara vokal utama.
    # Jika di ruangan pameran sangat berisik, angka 0.02 bisa kamu naikkan jadi 0.03 atau 0.05
    noise_threshold = 0.02 
    chunk[np.abs(chunk) < noise_threshold] = 0

    # ====================================================
    # 2. EKSTRAKSI MEL-SPECTROGRAM
    # ====================================================
    mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=8000)
    
    # C. Konversi ke dB dengan pembatasan dinamis (top_db)
    # top_db=60 berarti suara yang 60dB lebih pelan dari vokal akan dihitamkan total.
    # Ini sangat ampuh membuat background noise menjadi warna hitam/gelap di gambar.
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=60)

    # ====================================================
    # 3. GAMBAR SECARA VIRTUAL (IDENTIK DENGAN TRAINING)
    # ====================================================
    fig = plt.figure(figsize=(1, 1), dpi=128) # Menghasilkan 128x128
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    librosa.display.specshow(mel_db, sr=sr, fmax=8000, cmap='magma', ax=ax)

    # 4. Simpan ke Buffer (RAM) bukan ke Harddisk
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    # ====================================================
    # 5. CONVERT KEMBALI KE ARRAY UNTUK INPUT CNN
    # ====================================================
    img = Image.open(buf).convert('RGB')
    img = img.resize(image_size)
    arr = np.asarray(img).astype('float32') / 255.0 # Normalisasi 0-1
    
    return np.expand_dims(arr, axis=0), arr