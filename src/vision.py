import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import utils
import config
import cv2
import numpy as np
import pyautogui
import os
from PIL import Image

class Vision:
    def __init__(self, emulator=None):
        """
        Initialize the LLaVA model and processor.
        This model is used to extract a detailed description of the Pokémon Blue game state.
        
        Args:
            emulator: Optional Emulator instance to use for screen capture
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(config.LLAVA_MODEL_PATH)
        # Let Accelerate manage device placement (no manual .to(self.device) on model)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            config.LLAVA_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        utils.log(f"Vision model initialized on: {self.device}")
        
        # Store reference to emulator for screen capture
        self.emulator = emulator
        
    def set_emulator(self, emulator):
        """Set the emulator instance to use for screen capture"""
        self.emulator = emulator
        utils.log("Vision module linked to emulator for screen capture")

    def capture_screen(self):
        """
        Capture the current screen of the mGBA emulator.
        Returns a PIL Image in RGB format.
        
        If an emulator instance is available, uses its capture method.
        """
        if self.emulator:
            # Use the emulator's capture method which has automatic window detection
            frame = self.emulator.capture_screen()
            # Convert OpenCV BGR format to PIL RGB format
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            utils.log("WARNING: No emulator instance available for capture. Vision may not work correctly.")
            # Fallback to a blank image if no emulator is available
            image = Image.new('RGB', (480, 320), color=(0, 0, 0))
        
        # Save debug image
        debug_path = os.path.join(config.FRAME_SAVE_DIR, "debug_screen.png")
        image.save(debug_path)
        utils.log(f"Saved emulator screenshot: {debug_path}")
        return image

    def get_game_state(self):
        """
        Use LLaVA to analyze the captured screen and return a detailed description of the current game state.
        The description is crafted to capture important Pokémon Blue details:
          - Visible on-screen text (e.g., dialogue, menu options)
          - Whether a menu is open (main menu, party menu, inventory, battle menu, etc.)
          - Overworld information: player position, obstacles (walls, ledges), available movement options
          - Battle context: if in battle, what type (wild Pokémon vs. trainer), which Pokémon are displayed, etc.
          - Additional contextual cues (e.g., "Party Menu: Pikachu (HP 23/35), Bulbasaur (Full)", "In Shop", etc.)
        """
        image = self.capture_screen()

        # If the title screen is detected, we simply return that state.
        # (Assuming LLaVA itself will extract text; if the title screen is up, it will include phrases like "Pokémon Blue Version" or "Press Start".)
        # We rely on LLaVA to capture this in its output.
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": """
Analyze the following Pokémon Blue screen and provide a detailed, unambiguous description of the game state.
Consider the following aspects:
1. **Screen Text:** List all visible text (dialogue, menu headers, option labels).
2. **Location & Movement:** Describe where the player is (e.g., "in the center of Viridian City", "top-left corner of the overworld") and list available movement directions (e.g., "can move UP and LEFT, but blocked on the RIGHT").
3. **Menu State:** Indicate if a menu is open and, if so, specify which menu (e.g., "Party Menu", "Item Inventory", "Battle Menu"). List key options visible.
4. **Battle Context:** If in battle, specify whether it is a wild Pokémon battle or a trainer battle, and note the names/HP of the Pokémon if visible.
5. **Other Cues:** Note if the player is near an NPC, facing a wall or obstacle, or in an area that requires puzzle solving.
Provide your description in plain, clear language. Do not include any extra commentary or formatting tags.
                    """}
                ]
            }
        ]

        # Generate a prompt using the chat template (the processor will format it appropriately)
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        # Process both the image and the text prompt
        inputs = self.processor(
            text=text_prompt,
            images=image,
            return_tensors="pt"
        )
        # Move inputs to the device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=150
            )
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        utils.log(f"LLaVA Raw Output: {response}")
        return response