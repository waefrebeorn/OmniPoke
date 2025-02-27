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
    Updated Moondream-based vision & decision class.
    This class captures the screen from the emulator and uses the Moondream model
    to choose the next button press based on the current game state.
    It also accepts an optional task_instruction from Llama.
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

        # Valid button actions
        self.valid_buttons = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]

        # Base Moondream prompt with clear instructions
        self.base_prompt = """
Describe this Pokémon Blue game screen in detail. Focus on:

1. Game state (title screen, dialogue, battle, menu, or overworld)
2. Any visible text on screen
3. Character positions and movement options
4. Menu options if present
5. Battle information if in a battle

Be specific and accurate. Avoid general descriptions.
"""

        self.consecutive_title_screens = 0
        self.frames_captured = 0

    def capture_screen(self):
        """
        Capture the emulator screen as a PIL image.
        """
        if not self.emulator:
            utils.log("WARNING: No emulator provided to Vision. Returning blank image.")
            return Image.new('RGB', (160, 144), color=(0, 0, 0))
        
        self.frames_captured += 1
        frame = self.emulator.capture_screen()
        
        if self.frames_captured % 10 == 0:
            utils.save_frame(frame, f"frame_{self.frames_captured}.png")
            
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    def get_game_state_text(self):
        """
        Return a descriptive text about the current screen.
        """
        image = self.capture_screen()
        encoded = self.model.encode_image(image)
        detailed_prompt = self.base_prompt
        caption = self.model.query(encoded, detailed_prompt)["answer"]
        utils.log(f"Moondream Screen Description: {caption}")
        
        if "title" in caption.lower() or "blue version" in caption.lower():
            self.consecutive_title_screens += 1
            utils.log(f"Title screen detection count: {self.consecutive_title_screens}")
        else:
            self.consecutive_title_screens = 0
            
        return caption

    def get_next_action(self, task_instruction=None):
        """
        Use the Moondream model to choose the next button press.
        Instead of defaulting, this function continuously adjusts generation settings
        until a valid button press is extracted.
        """
        attempt = 0
        if self.consecutive_title_screens >= 2:
            utils.log("Multiple title screens detected. Forcing START button.")
            return "START"
            
        if task_instruction:
            prompt = f"""
You are playing Pokémon Blue. Look at the current game screen and follow this instruction:

{task_instruction}

Choose ONE of these buttons to press: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT

Reply with ONLY the button name in capital letters.
"""
        else:
            prompt = """
You are playing Pokémon Blue. Look at the current game screen.
If you see the title screen with "Blue Version", press START.
If you see text or dialogue, press A to continue.
If you're in a menu, use UP/DOWN to navigate and A to select.
If you're in the overworld, use UP/DOWN/LEFT/RIGHT to move.

Choose ONE of these buttons to press: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT

Reply with ONLY the button name in capital letters.
"""
        base_temperature = 0.7
        base_max_tokens = 20
        while True:
            curr_temp = base_temperature + (0.1 * (attempt % 10))
            curr_max_tokens = base_max_tokens + (2 * (attempt // 10))
            response = self.model.query(self.model.encode_image(self.capture_screen()), prompt, temperature=curr_temp, max_new_tokens=curr_max_tokens)["answer"]
            utils.log(f"Attempt {attempt+1} (temp={curr_temp}, max_tokens={curr_max_tokens}): Moondream raw decision: {response}")
            action = response.strip().upper()
            for button in self.valid_buttons:
                if button in action:
                    utils.log(f"Extracted button '{button}' from '{action}'")
                    return button
                    
            if "TITLE" in action or "BLUE VERSION" in action:
                utils.log("Title screen text detected in response, overriding to START")
                return "START"
            
            utils.log(f"Attempt {attempt+1}: No valid button in '{action}'. Retrying with new generation settings...")
            attempt += 1
