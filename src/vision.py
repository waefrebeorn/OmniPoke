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

        # Improved Moondream prompt with clear instructions
        self.moondream_prompt = """
You are playing Pokémon Blue on a Game Boy. Your goal is to complete the game as efficiently as possible by selecting the best button press for the current situation.

### Game Controls:
- UP, DOWN, LEFT, RIGHT → Move the player character or navigate menus.
- A → Confirm selections, talk to NPCs, progress dialogue.
- B → Cancel selections, close menus, or speed up text if needed.
- START → Start the game from the title screen or open the main menu.
- SELECT → Open secondary menus in specific cases.

### Decision Strategy:
1. **Title Screen:**  
   If the screen shows "Blue Version" or the Pokémon logo, PRESS START. Do NOT press A.
2. **Dialogue/Reading Text:**  
   If text is present, press A to continue (unless you are in a menu).
3. **Menus:**  
   Use UP/DOWN to navigate and A to confirm.
4. **Battles:**  
   Use directional buttons (UP/DOWN/LEFT/RIGHT) to choose moves or switch options and confirm with A.
5. **Overworld Movement:**  
   Use directional buttons to move around.

### IMPORTANT:
Return ONLY a single valid action (UP, DOWN, LEFT, RIGHT, A, B, START, SELECT) with no extra text.
Do not output any additional words.
"""

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
        Optional: Return a descriptive text about the current screen for debugging/logging.
        """
        image = self.capture_screen()
        encoded = self.model.encode_image(image)
        caption = self.model.caption(encoded, length="normal")["caption"]
        utils.log(f"Moondream Screen Caption: {caption}")
        return caption

    def get_next_action(self):
        """
        Use the Moondream model to choose the next button press.
        If the response is invalid, post-process and retry until a valid action is returned.
        """
        while True:
            image = self.capture_screen()
            encoded = self.model.encode_image(image)
            
            response = self.model.query(encoded, self.moondream_prompt)["answer"]
            utils.log(f"Moondream Raw Decision: {response}")
            
            # Clean and uppercase the response
            action = response.strip().upper()
            
            # Post-process: if Moondream outputs "BLUE VERSION", override to "START"
            if "BLUE VERSION" in action:
                utils.log("Detected 'BLUE VERSION' in response, overriding decision to 'START'.")
                action = "START"
            
            # Attempt to map partial matches if the action is not directly valid.
            if action not in self.valid_buttons:
                for button in self.valid_buttons:
                    if button in action:
                        utils.log(f"Mapping partial match '{action}' to '{button}'.")
                        action = button
                        break
            
            if action in self.valid_buttons:
                return action
            else:
                utils.log(f"Invalid Moondream action '{action}'. Retrying...")
                continue
