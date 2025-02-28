import time
import utils
import config
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2
import numpy as np
import os
from PIL import Image

class Vision:
    """
    Optimized Moondream-based vision class.
    This class captures the screen from the emulator and uses the Moondream model
    to analyze the game state.
    """
    def __init__(self, emulator=None):
        self.emulator = emulator
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load Moondream model with full float16 precision
        self.model = AutoModelForCausalLM.from_pretrained(
            config.VISION_MODEL_PATH,
            revision="2025-01-09",
            trust_remote_code=True,
            device_map={"": self.device},
            torch_dtype=torch.float16  # Use full FP16 precision
        )
        
        utils.log(f"Moondream model initialized on: {self.device} with full float16 precision")

        # Valid button actions
        self.valid_buttons = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]

        # Optimized Moondream prompt - shorter but still effective
        self.base_prompt = """
Describe this Pokémon Blue game screen. Focus on:
1. Game state (title screen, dialogue, battle, menu, or overworld)
2. Visible text or menu options
3. Character positions
4. Battle information if present
Be concise and specific.
"""

        # Image caching to avoid reprocessing identical frames
        self.image_cache = {}
        self.cache_limit = 5
        
        self.consecutive_title_screens = 0
        self.frames_captured = 0
        self.last_frame_hash = None

    def capture_screen(self):
        """
        Capture the emulator screen as a PIL image.
        """
        if not self.emulator:
            utils.log("WARNING: No emulator provided to Vision. Returning blank image.")
            return Image.new('RGB', (160, 144), color=(0, 0, 0))
        
        self.frames_captured += 1
        frame = self.emulator.capture_screen()
        
        # Save only every Nth frame to reduce disk I/O
        if self.frames_captured % config.FRAME_LOGGING_FREQUENCY == 0:
            utils.save_frame(frame, f"frame_{self.frames_captured}.png")
            
        # Convert BGR (OpenCV) to RGB and return PIL image
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    def _hash_image(self, image):
        """Create a simple hash for image caching."""
        # Convert to small grayscale for faster hashing
        img_small = image.resize((32, 32)).convert('L')
        img_array = np.array(img_small)
        return hash(img_array.tobytes())
    
    def get_game_state_text(self):
        """
        Return a descriptive text about the current screen with caching optimization.
        """
        image = self.capture_screen()
        image_hash = self._hash_image(image)
        
        # Check if this frame is in cache
        if image_hash in self.image_cache:
            utils.log("Using cached frame description")
            return self.image_cache[image_hash]
            
        # Check if this is the same as last frame
        if image_hash == self.last_frame_hash:
            utils.log("Frame unchanged, reusing last description")
            return self.image_cache.get(image_hash, "Game screen with no changes")
        
        self.last_frame_hash = image_hash
        
        # Generate caption using Moondream
        with torch.inference_mode():
            caption = self.model.query(
                image,
                self.base_prompt,
            )["answer"]
            
        utils.log(f"Moondream Screen Description: {caption}")
        
        # Update cache
        if len(self.image_cache) >= self.cache_limit:
            oldest = next(iter(self.image_cache))
            del self.image_cache[oldest]
        self.image_cache[image_hash] = caption
        
        # Track title screen occurrences
        if "title" in caption.lower() or "blue version" in caption.lower():
            self.consecutive_title_screens += 1
            utils.log(f"Title screen detection count: {self.consecutive_title_screens}")
        else:
            self.consecutive_title_screens = 0
            
        return caption

    def get_next_action(self, task_instruction=None):
        """
        Use the Moondream model to choose the next button press.
        Optimized with fast paths and more efficient generation settings.
        """
        # Title screen fast path
        if self.consecutive_title_screens >= 2:
            utils.log("Multiple title screens detected. Forcing START button.")
            return "START"
        
        image = self.capture_screen()
        
        # Use task instruction if provided, otherwise use default prompt
        if task_instruction:
            prompt = f"""
You are playing Pokémon Blue. Choose a button based on this instruction:
{task_instruction}
Reply with ONLY one button: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT
"""
        else:
            prompt = """
You are playing Pokémon Blue. Choose ONE button to press:
If you see title screen: START
If you see text/dialogue: A
If in menu: UP/DOWN to navigate, A to select
If in overworld: UP/DOWN/LEFT/RIGHT to move

Reply with ONLY one button: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT
"""
        # Generate with optimized settings
        with torch.inference_mode():
            response = self.model.query(
                image,
                prompt,
            )["answer"]
            
        utils.log(f"Moondream raw decision: {response}")
        action = response.strip().upper()
        
        # Extract button by direct pattern matching
        for button in self.valid_buttons:
            if button in action:
                utils.log(f"Extracted button '{button}' from '{action}'")
                return button
                
        # Title screen override
        if "TITLE" in action or "BLUE VERSION" in action:
            utils.log("Title screen text detected in response, overriding to START")
            return "START"
        
        # Default to "redo" if no valid button found - NEEDED FOR AI PLAYTHROUGH
        utils.log(f"No valid button in '{action}'. Defaulting to RETRYING.")
        return self.get_next_action()  # Recursive call to try again