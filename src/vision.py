import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import config
import utils
import cv2
import numpy as np
import os
from PIL import Image

class Vision:
    """
    Moondream-based vision & decision. One class that:
      1. Captures the screen from the emulator.
      2. Uses Moondream to pick the next button to press.
    """
    def __init__(self, emulator=None):
        self.emulator = emulator
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load Moondream
        self.processor = AutoProcessor.from_pretrained(
            config.VISION_MODEL_PATH,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.VISION_MODEL_PATH,
            revision="2025-01-09",
            trust_remote_code=True,
            device_map={"": self.device}
        )
        utils.log(f"Moondream model initialized on: {self.device}")

        # Valid buttons
        self.valid_buttons = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]

    def capture_screen(self):
        """
        Capture the emulator screen as a PIL image.
        """
        if not self.emulator:
            utils.log("WARNING: No emulator provided to Vision. Returning blank image.")
            return Image.new('RGB', (160, 144), color=(0, 0, 0))

        frame = self.emulator.capture_screen()
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def get_game_state_text(self):
        """
        Optional: Return a descriptive text about the current screen, for debugging or logs.
        """
        image = self.capture_screen()
        encoded = self.model.encode_image(image)

        # Basic caption from Moondream
        caption = self.model.caption(encoded, length="normal")["caption"]
        utils.log(f"Moondream Screen Caption: {caption}")
        return caption

    def get_next_action(self):
        """
        Use Moondream to directly choose the next button press from [UP, DOWN, LEFT, RIGHT, A, B, START, SELECT].
        We'll retry infinitely if Moondream doesn't produce a valid single action.
        """
        while True:
            image = self.capture_screen()
            encoded = self.model.encode_image(image)

            # Improved prompt: if on title screen or in a menu, the best action is to press START.
            moondream_prompt = f"""
You are playing Pokémon Blue on a Game Boy.
Analyze this screenshot carefully. If the screen shows the title screen or a menu, the best action is to press START to begin or continue the game.
Otherwise, decide the single best button to press from the following options: {", ".join(self.valid_buttons)}.
Return ONLY one of these words (UP, DOWN, LEFT, RIGHT, A, B, START, SELECT) with no extra text.
"""
            response = self.model.query(encoded, moondream_prompt)["answer"]
            utils.log(f"Moondream Raw Decision: {response}")

            # Clean up the response, uppercase it
            action = response.strip().upper()

            # If the action is valid, return it. Otherwise, retry.
            if action in self.valid_buttons:
                return action
            else:
                utils.log(f"Invalid Moondream action '{action}'. Retrying...")
