import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import utils
import config
import cv2
import numpy as np
import os
from PIL import Image

class Vision:
    def __init__(self, emulator=None):
        """
        Initialize the Moondream model for vision-based game state extraction.
        Moondream extracts detailed game information from Pokémon Blue screenshots.
        
        Args:
            emulator: Optional Emulator instance for screen capture.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ✅ Fix: Trust remote code to resolve loading issues
        self.processor = AutoProcessor.from_pretrained(
            config.VISION_MODEL_PATH,
            trust_remote_code=True  # Required to allow execution of Moondream’s custom code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.VISION_MODEL_PATH,
            revision="2025-01-09",
            trust_remote_code=True,  # Required to execute Moondream's model code
            device_map={"": self.device}  # Auto-assign to GPU if available
        )
        utils.log(f"Moondream vision model initialized on: {self.device}")
        
        self.emulator = emulator  # Store emulator reference

    def capture_screen(self):
        """
        Capture the Game Boy emulator screen.
        Returns a PIL Image.
        """
        if self.emulator:
            frame = self.emulator.capture_screen()
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            utils.log("WARNING: No emulator available for screen capture.")
            return Image.new('RGB', (480, 320), color=(0, 0, 0))

    def get_game_state(self):
        """
        Use Moondream to analyze the captured screen and extract game state information.
        """
        image = self.capture_screen()
        encoded_image = self.model.encode_image(image)

        # Extract all visible on-screen text (acts as OCR)
        screen_text = self.model.caption(encoded_image, length="normal")["caption"]
        utils.log(f"Extracted Screen Text: {screen_text}")

        # Advanced prompt engineering for game state analysis
        moondream_prompt = """
        Analyze this Pokémon Blue game screen and provide a structured breakdown of the game state.
        
        1. **Screen Text:** List all visible dialogue, menu options, and in-game messages.
        2. **Player Location:** Describe where the player is as a general screen region and describe the screen scenery.
        3. **Available Movement:** List available movement directions, this is an rpg game so sometimes you are in a menu and cannot move.
        4. **Menu State:** Identify if a menu is open (Party, Inventory, Battle, Shop, etc.), and list key visible options.
        5. **Battle Context:** If a battle is happening, determine if it’s wild or a trainer battle. List Pokémon names, HP, and battle status.
        6. **Nearby Objects & NPCs:** Identify visible NPCs, obstacles, or interactable objects.
        7. **Logical Next Action:** Suggest the most optimal button press for progressing in the game.
        
        Provide a concise, structured response without unnecessary commentary.
        """

        # Query Moondream with the enhanced prompt
        game_state_description = self.model.query(encoded_image, moondream_prompt)["answer"]
        utils.log(f"Moondream Game State Analysis: {game_state_description}")

        # Combine screen text with extracted game state
        full_game_state = f"{screen_text}\n\n{game_state_description}"
        return full_game_state
