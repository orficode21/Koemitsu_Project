from pathlib import Path
import librosa

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / 'src'
ASSETS_DIR = BASE_DIR / 'assets'
GUIDES_DIR = ASSETS_DIR / 'guides'
GUIDE_NOTES_DIR = GUIDES_DIR / 'notes'
GUIDE_DEGREES_DIR = GUIDES_DIR / 'degrees'
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
UPLOADS_DIR = DATA_DIR / 'uploads'
LOGS_DIR = DATA_DIR / 'logs'

MODEL_CANDIDATES = [
    #MODELS_DIR / 'vocal_model_loso.h5',
    MODELS_DIR / 'vocal_model_loso2.h5',
    #BASE_DIR / 'vocal_model_loso.h5',
    BASE_DIR / 'vocal_model_loso2.h5',
]

SR = 22050
PYIN_FMIN = librosa.note_to_hz('C2')
PYIN_FMAX = librosa.note_to_hz('C6')

SLICE_WIN = 1.0
SLICE_HOP = 0.5
SAMPLES_SLICE_WIN = int(SR * SLICE_WIN)
SAMPLES_SLICE_HOP = int(SR * SLICE_HOP)

MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11, 12]
DEGREE_LABELS = ['Do', 'Re', 'Mi', 'Fa', 'Sol', 'La', 'Si', 'Do tinggi']
DEGREE_SLUGS = ['do', 're', 'mi', 'fa', 'sol', 'la', 'si', 'do_high']
DEGREE_TO_OFFSET = {
    'Do': 0,
    'Re': 2,
    'Mi': 4,
    'Fa': 5,
    'Sol': 7,
    'La': 9,
    'Si': 11,
}

DEFAULT_RES_THRESHOLD = 0.55
DEFAULT_PITCH_TOLERANCE_CENTS = 150.0
DEFAULT_COMFORT_ROLE = 'Do'

GUIDE_TONE_DURATION = 1.6
GUIDE_TONE_GAIN = 0.18
GUIDE_SAMPLE_RATE = SR

CALIBRATION_MIN_VOICED_RATIO = 0.25
TEST_MIN_VOICED_RATIO = 0.20

AUDIO_FILE_TYPES = ['wav', 'mp3', 'ogg', 'm4a', 'flac']

CALIBRATION_GUIDE_CANDIDATES = [
    GUIDES_DIR / 'calibration_aaa.wav',
    GUIDES_DIR / 'calibration.wav',
    GUIDES_DIR / 'calibration_example.wav',
]
