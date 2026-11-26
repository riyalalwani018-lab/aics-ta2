from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "pixi_efficientnetb0_best.keras"

IMAGE_SIZE = (260, 260)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB
