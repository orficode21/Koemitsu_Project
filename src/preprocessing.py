import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
from PIL import Image

def preprocess_chunk_to_rgb_magma(chunk, sr=22050, image_size=(128, 128)):
    # 1. Ekstrak Mel-Spectrogram
    mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 2. Gambar secara virtual (SAMA PERSIS DENGAN KODE DATASET GENERATOR KAMU)
    fig = plt.figure(figsize=(1, 1), dpi=128) # Menghasilkan 128x128
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    librosa.display.specshow(mel_db, sr=sr, fmax=8000, cmap='magma', ax=ax)

    # 3. Simpan ke Buffer (RAM) bukan ke Harddisk
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    # 4. Convert kembali ke Array untuk CNN
    img = Image.open(buf).convert('RGB')
    img = img.resize(image_size)
    arr = np.asarray(img).astype('float32') / 255.0 # Normalisasi 0-1
    
    return np.expand_dims(arr, axis=0), arr