import os
import cv2
import datetime
import config

def log(message):
    """Logs a message with a timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    log_file = os.path.join(config.LOG_DIR, "log.txt")
    with open(log_file, "a") as f:
        f.write(log_message + "\n")

def save_frame(frame, filename=None):
    """Saves a frame for debugging if enabled."""
    if config.SAVE_FRAMES:
        if filename is None:
            filename = f"frame_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(config.FRAME_SAVE_DIR, filename)
        cv2.imwrite(filepath, frame)
        log(f"Saved frame: {filepath}")

def map_action_to_key(action):
    """Maps an action string to the emulator's keybinds."""
    return config.KEY_MAPPING.get(action.upper(), None)
