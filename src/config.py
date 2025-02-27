import os

# === LLaMA Settings ===
LLAMA_MODEL_PATH = "meta-llama/Llama-3.2-3B"  # 3B model path
LLAMA_TEMPERATURE = 0.7  # Lower = more deterministic
LLAMA_MAX_TOKENS = 80  # Increased for better decision-making
LLAMA_TOP_K = 5  # Limits output randomness

# === LLaVA Vision Model ===
LLAVA_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"  # Using LLaVA

# === ROM & Emulator Settings ===
ROM_PATH = "roms/pokemon_blue.gb"
POLLING_INTERVAL = 1  # Default polling interval in seconds
EMULATOR_PATH = "C:\\Program Files\\mGBA\\mGBA.exe"

# Screenshot location (same as ROM directory)
SCREENSHOT_PATH = os.path.join(os.path.dirname(ROM_PATH), "screenshot.png")

# === Debugging & Logging ===
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure log directory exists
SAVE_FRAMES = True
FRAME_SAVE_DIR = os.path.join(os.getcwd(), "logs", "frames")
os.makedirs(FRAME_SAVE_DIR, exist_ok=True)

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
