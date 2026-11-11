"""FruitSenseAI configuration module.

Defines model checkpoints, paths, class names, and API configurations.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load .env file from project root if it exists
load_dotenv(PROJECT_ROOT / ".env")

# Model checkpoints and directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_WEB_DIR = OUTPUT_DIR / "web_runs"
OUTPUT_WEB_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CHECKPOINT_PATH = (
    OUTPUT_DIR
    / "six_model_comparison"
    / "checkpoints"
    / "MobileNetV3Large_best.pth"
)

# Canonical 6 classes from training
CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
]

# Human-readable labels
CLASS_DISPLAY_NAMES = {
    "freshapples": "Fresh Apple",
    "freshbanana": "Fresh Banana",
    "freshoranges": "Fresh Orange",
    "rottenapples": "Rotten Apple",
    "rottenbanana": "Rotten Banana",
    "rottenoranges": "Rotten Orange",
}


def display_class_name(class_name: str) -> str:
    """Create a readable label for any ImageFolder class name."""
    if class_name in CLASS_DISPLAY_NAMES:
        return CLASS_DISPLAY_NAMES[class_name]
    label = class_name.replace("_", " ").replace("-", " ").strip()
    label = label.replace("fresh", "Fresh ", 1) if label.lower().startswith("fresh") else label
    label = label.replace("rotten", "Rotten ", 1) if label.lower().startswith("rotten") else label
    return " ".join(label.split()).title()


def fruit_family_name(class_name: str) -> str:
    """Return the fruit portion of a freshness class, including new fruits."""
    name = class_name.lower().replace("_", " ").replace("-", " ").strip()
    for condition in ("fresh", "rotten"):
        if name.startswith(condition):
            name = name[len(condition):].strip()
            break
    return name or "unknown"

# Image parameters
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MAX_UPLOAD_SIZE_MB = 15
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# VLM configuration
VLM_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
VLM_MAX_NEW_TOKENS = 48

# Gemini configuration
# Default to gemini-2.5-flash which is fast, cost-effective, and officially supported
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Sarvam TTS configuration
SARVAM_TTS_MODEL = "bulbul:v4-flash"
SUPPORTED_LANGUAGES = {
    "en-IN": "English (India)",
    "ta-IN": "Tamil (தமிழ்)",
    "hi-IN": "Hindi (हिन्दी)",
}

def get_gemini_api_key() -> str | None:
    """Return the Gemini API key from environment, checking GEMINI_API_KEY then GOOGLE_API_KEY."""
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

def get_sarvam_api_key() -> str | None:
    """Return the Sarvam AI API key from environment."""
    return os.getenv("SARVAM_API_KEY")
