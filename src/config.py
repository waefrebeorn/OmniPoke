import os

# === Vision Model Settings ===
VISION_MODEL_PATH = "vikhyatk/moondream2"

# === Memory Optimization Settings ===
# Enable memory optimizations (8-bit quantization and offloading)
USE_8BIT_QUANTIZATION = True
USE_OFFLOADING = True

# === ROM & Emulator Settings ===
ROM_PATH = "roms/pokemon_blue.gb"
POLLING_INTERVAL = 0.8  # Reduced polling interval for faster gameplay
EMULATOR_PATH = "C:\\Program Files\\mGBA\\mGBA.exe"

# Screenshot location (same as ROM directory)
SCREENSHOT_PATH = os.path.join(os.path.dirname(ROM_PATH), "screenshot.png")

# === Debugging & Logging ===
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure log directory exists
SAVE_FRAMES = True
FRAME_SAVE_DIR = os.path.join(os.getcwd(), "logs", "frames")
os.makedirs(FRAME_SAVE_DIR, exist_ok=True)

# Reduce frame logging frequency to improve performance
FRAME_LOGGING_FREQUENCY = 20  # Save only every Nth frame

# === Cache Settings ===
# Enhanced caching to reduce model calls
VISION_CACHE_SIZE = 10
DECISION_CACHE_SIZE = 20
ENABLE_SIMILARITY_CACHE = True  # Enable fuzzy matching for cache hits

# === Input Settings ===
KEY_MAPPING = {
    "UP": "up",
    "DOWN": "down",
    "LEFT": "left",
    "RIGHT": "right",
    "A": "z",      # Adjust based on emulator key mapping
    "B": "x",
    "START": "enter",
    "SELECT": "backspace",
    "SCREENSHOT": "F12"  # New key for triggering mGBA screenshot
}

# === Performance Settings ===
# Performance optimization flags
TORCH_COMPILE = True      # Use torch.compile for models if available
HALF_PRECISION = True     # Use FP16 precision
LOW_CPU_USAGE = True      # Enable low CPU memory usage