import time
import utils
import config
import torch
from transformers import AutoModelForCausalLM
import cv2
import numpy as np
import os
from PIL import Image

class Vision:
    """
    Optimized Moondream-based vision class with enhanced game state detection capabilities.
    Uses brightness checks to distinguish between deep dark caves and walking areas,
    and leverages a walkthrough-based context for improved environment consistency.
    """
    def __init__(self, emulator=None):
        self.emulator = emulator
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load Moondream model with full float16 precision for compatibility
        self.model = AutoModelForCausalLM.from_pretrained(
            config.VISION_MODEL_PATH,
            revision="2025-01-09",
            trust_remote_code=True,
            device_map={"": self.device},
            torch_dtype=torch.float16
        )
        if torch.cuda.is_available() and config.TORCH_COMPILE:
            self.model = torch.compile(self.model)
        
        utils.log(f"Moondream model initialized on: {self.device} with full float16 precision")

        self.valid_buttons = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]

        self.base_prompt = """
Describe this Pokémon Blue game screen in detail. Focus on:
1. Game state (be specific: title screen, dialogue, battle type, menu, overworld, cave, dark cave, gym)
2. All visible text or menu options
3. Character and NPC positions
4. Environment details (indoors, outdoors, cave, water)
5. Battle information if present (Pokémon names, HP, moves)
6. Any obstacles or interactive elements (doors, signs, trees, boulders)

Be comprehensive but concise. Include environmental cues that indicate location type.
"""

        self.specialized_prompts = {
            "CAVE_DETECTION": """
Is the player in a cave environment? Look for:
- Dark/rocky surroundings
- Limited visibility (but not pitch black)
- Cave entrance/exit and open areas
- Rock formations and tunnels
Describe the cave features and any visible paths or items.
""",
            "DARK_CAVE_DETECTION": """
Is this a dark cave where Flash is needed? Look for:
- Very limited visibility (small area around the player)
- Mostly black or near-black surroundings
- Difficulty distinguishing paths or walls
Describe the limited visibility and any cues.
"""
        }

        self.image_cache = {}
        self.cache_limit = 10
        self.consecutive_title_screens = 0
        self.frames_captured = 0
        self.last_frame_hash = None
        self.current_environment = "UNKNOWN"
        self.environment_confidence = 0
        self.environment_history = []
        self.max_history = 5

        # Save last confirmed cave state for continuity.
        self.last_cave_state = None

        # Walkthrough-based planned states (example subset extracted from the walkthrough)
        # You can expand this list to cover more detailed segments.
        self.walkthrough_path = [
            "PALLET_TOWN", "ROUTE_1", "VIRIDIAN_CITY", "POKEMON_CENTER",
            "OAKS_LAB", "VIRIDIAN_FOREST", "MT_MOON", "PEWTER_CITY"
        ]

    def capture_screen(self):
        if not self.emulator:
            utils.log("WARNING: No emulator provided to Vision. Returning blank image.")
            return Image.new('RGB', (160, 144), color=(0, 0, 0))
        
        self.frames_captured += 1
        frame = self.emulator.capture_screen()
        
        if self.frames_captured % config.FRAME_LOGGING_FREQUENCY == 0:
            utils.save_frame(frame, f"frame_{self.frames_captured}.png")
            
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    def _hash_image(self, image):
        img_small = image.resize((32, 32)).convert('L')
        img_array = np.array(img_small)
        return hash(img_array.tobytes())
    
    def get_average_brightness(self, image):
        grayscale = image.convert("L")
        np_image = np.array(grayscale)
        return np.mean(np_image)
    
    def detect_environment(self, description):
        description_upper = description.upper()
        if any(term in description_upper for term in ["DARK CAVE", "CAN'T SEE", "LIMITED VISIBILITY", "FLASH NEEDED"]):
            return "DARK_CAVE"
        if any(term in description_upper for term in ["MT MOON", "MT. MOON"]):
            return "MT_MOON"
        if "ROCK TUNNEL" in description_upper:
            return "ROCK_TUNNEL"
        if "VICTORY ROAD" in description_upper:
            return "VICTORY_ROAD"
        if any(term in description_upper for term in ["CAVE", "TUNNEL", "UNDERGROUND", "ROCKY"]):
            return "CAVE"
        if any(term in description_upper for term in ["POKEMON CENTER", "POKÉMON CENTER", "NURSE JOY", "HEALING"]):
            return "POKEMON_CENTER"
        if any(term in description_upper for term in ["MART", "SHOP", "STORE", "CLERK", "BUY"]):
            return "POKEMON_MART"
        if "GYM" in description_upper:
            return "GYM"
        if any(term in description_upper for term in ["HOUSE", "BUILDING", "INSIDE", "INTERIOR", "INDOOR"]):
            return "BUILDING"
        if "GYM LEADER" in description_upper and "BATTLE" in description_upper:
            return "GYM_LEADER_BATTLE"
        if "TRAINER" in description_upper and "BATTLE" in description_upper:
            return "TRAINER_BATTLE"
        if "WILD" in description_upper and "BATTLE" in description_upper:
            return "WILD_BATTLE"
        if any(term in description_upper for term in ["BATTLE", "FIGHT", "POKEMON VS", "POKÉMON VS"]):
            return "BATTLE"
        if any(term in description_upper for term in ["TITLE SCREEN", "BLUE VERSION", "PRESS START"]):
            return "TITLE_SCREEN"
        if any(term in description_upper for term in ["TEXT BOX", "DIALOGUE", "TALKING", "MESSAGE"]):
            return "DIALOGUE"
        if any(term in description_upper for term in ["MENU", "OPTIONS", "ITEMS", "POKEMON LIST", "POKÉMON LIST"]):
            return "MENU"
        return "OVERWORLD"
    
    def update_environment_tracking(self, detected_env):
        self.environment_history.append(detected_env)
        if len(self.environment_history) > self.max_history:
            self.environment_history.pop(0)
        env_counts = {}
        for env in self.environment_history:
            env_counts[env] = env_counts.get(env, 0) + 1
        most_common_env = max(env_counts.items(), key=lambda x: x[1])
        confidence = most_common_env[1] / len(self.environment_history)
        if confidence >= 0.6:
            prev_env = self.current_environment
            self.current_environment = most_common_env[0]
            self.environment_confidence = confidence
            if self.current_environment in ["CAVE", "DARK_CAVE"]:
                self.last_cave_state = self.current_environment
            if prev_env != self.current_environment:
                utils.log(f"Environment change: {prev_env} -> {self.current_environment} (confidence: {confidence:.2f})")
        return self.current_environment
    
    def get_game_state_text(self):
        image = self.capture_screen()
        image_hash = self._hash_image(image)
        
        brightness = self.get_average_brightness(image)
        # Define thresholds for dark and bright conditions.
        DARK_THRESHOLD = 40    # Below this value, consider the screen dark.
        BRIGHT_THRESHOLD = 60  # Above this, the screen is clearly bright.
        
        # If the screen is very dark, force a dark cave state.
        if brightness < DARK_THRESHOLD:
            utils.log(f"Dark screen detected (brightness: {brightness:.1f}). Forcing DARK_CAVE state.")
            caption = "Dark cave environment with very limited visibility. Flash needed."
            detected_env = "DARK_CAVE"
            self.update_environment_tracking(detected_env)
            self.last_cave_state = detected_env
            if len(self.image_cache) >= self.cache_limit:
                oldest = next(iter(self.image_cache))
                del self.image_cache[oldest]
            self.image_cache[image_hash] = caption
            return caption
        
        # Conversely, if brightness is high and we had a cave state, switch to a brighter cave.
        if brightness > BRIGHT_THRESHOLD and self.last_cave_state in ["DARK_CAVE", "CAVE"]:
            utils.log(f"Bright screen detected (brightness: {brightness:.1f}). Interpreting as walking through a cave.")
            # Override to "CAVE" rather than dark cave.
            self.last_cave_state = "CAVE"
        
        if image_hash in self.image_cache:
            cached_description = self.image_cache[image_hash]
            detected_env = self.detect_environment(cached_description)
            self.update_environment_tracking(detected_env)
            utils.log(f"Using cached frame description (environment: {self.current_environment})")
            return cached_description
            
        if image_hash == self.last_frame_hash:
            cached_description = self.image_cache.get(image_hash, "Game screen with no changes")
            utils.log("Frame unchanged, reusing last description")
            return cached_description
        
        self.last_frame_hash = image_hash
        
        # Choose prompt based on recent environment history and walkthrough context.
        prompt_to_use = self.base_prompt
        if any(env in ["CAVE", "MT_MOON", "ROCK_TUNNEL", "VICTORY ROAD"] for env in self.environment_history[-2:]):
            prompt_to_use = self.specialized_prompts["CAVE_DETECTION"]
            utils.log("Using specialized cave detection prompt")
        elif "DARK_CAVE" in self.environment_history[-2:]:
            prompt_to_use = self.specialized_prompts["DARK_CAVE_DETECTION"]
            utils.log("Using specialized dark cave detection prompt")
        
        with torch.inference_mode():
            caption = self.model.query(
                image,
                prompt_to_use,
            )["answer"]
            
        utils.log(f"Moondream Screen Description: {caption}")
        
        detected_env = self.detect_environment(caption)
        # If the description returns as overworld but our last cave context is active from walkthrough, force cave state.
        if detected_env == "OVERWORLD" and self.last_cave_state in ["CAVE", "DARK_CAVE"]:
            utils.log("Forcing cave state from previous context due to walkthrough continuity.")
            detected_env = self.last_cave_state
            caption = f"Continuing {detected_env} exploration."
        
        self.update_environment_tracking(detected_env)
        
        if len(self.image_cache) >= self.cache_limit:
            oldest = next(iter(self.image_cache))
            del self.image_cache[oldest]
        self.image_cache[image_hash] = caption
        
        if ("title screen" in caption.lower() or "blue version" in caption.lower()) and \
           ("professor" not in caption.lower() and "oak" not in caption.lower()):
            self.consecutive_title_screens += 1
            utils.log(f"Title screen detection count: {self.consecutive_title_screens}")
        else:
            self.consecutive_title_screens = 0
            
        return caption

    def get_context_aware_prompt(self, task_instruction=None):
        base_prefix = "You are playing Pokémon Blue. "
        base_suffix = "\nReply with ONLY one button: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT"
        env_instructions = {
            "DARK_CAVE": "You're in a dark cave with very limited visibility. Consider using Flash if available. ",
            "CAVE": "You're exploring a cave with visible details and open areas. Watch for wild Pokémon and items. ",
            "MT_MOON": "You're in Mt. Moon. Watch for Zubat encounters and fossil researchers. ",
            "ROCK_TUNNEL": "You're in Rock Tunnel. This cave is complex with many trainers. ",
            "VICTORY_ROAD": "You're in Victory Road. This is a challenging cave with strength puzzles. ",
            "POKEMON_CENTER": "You're in a Pokémon Center. Consider talking to Nurse Joy to heal your Pokémon. ",
            "POKEMON_MART": "You're in a Pokémon Mart. You can buy items from the clerk. ",
            "GYM": "You're in a Pokémon Gym. Navigate to the Gym Leader while defeating trainers. ",
            "WILD_BATTLE": "You're in a wild Pokémon battle. Choose fight for moves, or other options like Run. ",
            "TRAINER_BATTLE": "You're in a trainer battle. You must win to proceed. Choose your moves wisely. ",
            "GYM_LEADER_BATTLE": "You're battling a Gym Leader! This is an important battle for a badge. ",
            "TITLE_SCREEN": "You're at the title screen. ",
            "DIALOGUE": "You're in a dialogue or text screen. ",
            "MENU": "You're navigating a menu. Use directional buttons to navigate and A to select. ",
            "OVERWORLD": "You're exploring the overworld. "
        }
        
        env_instruction = env_instructions.get(self.current_environment, "Choose a button based on what you see. ")
        # Integrate walkthrough context if available.
        walkthrough_hint = ""
        if self.walkthrough_path:
            walkthrough_hint = "Remember your planned path from the walkthrough."
        if task_instruction:
            prompt = f"{base_prefix}{walkthrough_hint}\nChoose a button based on this instruction:\n{task_instruction}{base_suffix}"
        else:
            prompt = f"{base_prefix}{env_instruction} {walkthrough_hint}{base_suffix}"
            
        return prompt

    def get_next_action(self, task_instruction=None):
        if self.consecutive_title_screens >= 2:
            utils.log("Multiple title screens detected. Forcing START button.")
            return "START"
        
        image = self.capture_screen()
        prompt = self.get_context_aware_prompt(task_instruction)
        utils.log(f"Using prompt: {prompt}")
        
        with torch.inference_mode():
            response = self.model.query(
                image,
                prompt,
            )["answer"]
            
        utils.log(f"Moondream raw decision: {response}")
        action = response.strip().upper()
        
        if ("TITLE SCREEN" in action or "BLUE VERSION" in action) and \
           ("PROFESSOR" not in action and "OAK" not in action):
            utils.log("Title screen text detected in response, overriding to START")
            return "START"
        
        for button in self.valid_buttons:
            if button in action:
                utils.log(f"Extracted button '{button}' from '{action}'")
                return button
                
        if self.current_environment == "DARK_CAVE":
            utils.log("In dark cave with no clear action. Defaulting to START to access Flash.")
            return "START"
        
        if self.current_environment == "DIALOGUE":
            utils.log("In dialogue with no clear action. Defaulting to A.")
            return "A"
        
        utils.log(f"No valid button in '{action}'. Defaulting to RETRYING.")
        return self.get_next_action(task_instruction)
