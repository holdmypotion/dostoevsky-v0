import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "output"))
    TEMP_DIR = Path(BASE_DIR / "temp")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    OLLAMA_ENDPOINT = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
    OLLAMA_MODEL = os.getenv("MODEL_NAME", "mistral")
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", 500))

    TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
    AUDIO_SAMPLE_RATE = 22050
    SPEECH_LANGUAGE = os.getenv("SPEECH_LANGUAGE", "en")

    IMAGE_SIZE = (1080, 1080)
    IMAGE_MODEL = "stable-diffusion-v1-5"

    VIDEO_FPS = 30
    VIDEO_RESOLUTION = (1080, 1920)  # Instagram portrait
    TRANSITION_DURATION = 0.5 