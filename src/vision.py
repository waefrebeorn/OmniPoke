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
    This class captures the screen from the emulator and uses the Moondream model
    to analyze the game state with improved prompts for specialized areas like caves.
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
            torch_dtype=torch.float16  # Use full FP16 precision
        )
        
        # Optionally compile the model for faster inference if enabled in config
        if torch.cuda.is_available() and config.TORCH_COMPILE:
            self.model = torch.compile(self.model)
        
        utils.log(f"Moondream model initialized on: {self.device} with full float16 precision")

        # Valid button actions
        self.valid_buttons = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]

        # Enhanced base prompt for better game state detection
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

        # Environment-specific prompts for better detection
        self.specialized_prompts = {
            "CAVE_DETECTION": """
Is the player in a cave environment? Look for:
- Dark/rocky surroundings
- Limited visibility
- Cave entrance/exit
- Rock formations
- Multiple tunnel paths
Describe the cave features and any visible paths or items.
""",
            "DARK_CAVE_DETECTION": """
Is this a dark cave where Flash is needed? Look for:
- Very limited visibility (small visible area around player)
- Black surroundings
- Difficulty seeing paths or walls
Describe the visibility level and any visible elements.
"""
        }

        # Image caching to avoid reprocessing identical frames
        self.image_cache = {}
        self.cache_limit = 10  # Increased cache size
        
        self.consecutive_title_screens = 0
        self.frames_captured = 0
        self.last_frame_hash = None
        
        # Track environmental context across frames
        self.current_environment = "UNKNOWN"
        self.environment_confidence = 0
        self.environment_history = []  # Track last 5 environment detections
        self.max_history = 5

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
    
    def detect_environment(self, description):
        """
        Analyze the screen description to detect specific environments.
        Returns the detected environment type.
        """
        description_upper = description.upper()
        
        # Cave detection with priority on dark caves
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
            
        # Building interiors
        if any(term in description_upper for term in ["POKEMON CENTER", "POKÉMON CENTER", "NURSE JOY", "HEALING"]):
            return "POKEMON_CENTER"
        if any(term in description_upper for term in ["MART", "SHOP", "STORE", "CLERK", "BUY"]):
            return "POKEMON_MART"
        if "GYM" in description_upper:
            return "GYM"
        if any(term in description_upper for term in ["HOUSE", "BUILDING", "INSIDE", "INTERIOR", "INDOOR"]):
            return "BUILDING"
            
        # Battle types
        if "GYM LEADER" in description_upper and "BATTLE" in description_upper:
            return "GYM_LEADER_BATTLE"
        if "TRAINER" in description_upper and "BATTLE" in description_upper:
            return "TRAINER_BATTLE"
        if "WILD" in description_upper and "BATTLE" in description_upper:
            return "WILD_BATTLE"
        if any(term in description_upper for term in ["BATTLE", "FIGHT", "POKEMON VS", "POKÉMON VS"]):
            return "BATTLE"
            
        # Other specific states
        if any(term in description_upper for term in ["TITLE SCREEN", "BLUE VERSION", "PRESS START"]):
            return "TITLE_SCREEN"
        if any(term in description_upper for term in ["TEXT BOX", "DIALOGUE", "TALKING", "MESSAGE"]):
            return "DIALOGUE"
        if any(term in description_upper for term in ["MENU", "OPTIONS", "ITEMS", "POKEMON LIST", "POKÉMON LIST"]):
            return "MENU"
            
        # Default to overworld
        return "OVERWORLD"
    
    def update_environment_tracking(self, detected_env):
        """
        Update environment tracking based on new detection,
        using a history-based approach for stability.
        """
        # Add to history and maintain max size
        self.environment_history.append(detected_env)
        if len(self.environment_history) > self.max_history:
            self.environment_history.pop(0)
            
        # Count occurrences of each environment in history
        env_counts = {}
        for env in self.environment_history:
            env_counts[env] = env_counts.get(env, 0) + 1
            
        # Find most common environment in history
        most_common_env = max(env_counts.items(), key=lambda x: x[1])
        most_common_count = most_common_env[1]
        confidence = most_common_count / len(self.environment_history)
        
        # Update current environment if confidence is high enough
        if confidence >= 0.6:  # 60% threshold for changing environment
            prev_env = self.current_environment
            self.current_environment = most_common_env[0]
            self.environment_confidence = confidence
            
            if prev_env != self.current_environment:
                utils.log(f"Environment change: {prev_env} -> {self.current_environment} (confidence: {confidence:.2f})")
        
        return self.current_environment
    
    def get_game_state_text(self):
        """
        Return a descriptive text about the current screen with enhanced environment detection.
        """
        image = self.capture_screen()
        image_hash = self._hash_image(image)
        
        # Check if this frame is in cache
        if image_hash in self.image_cache:
            cached_description = self.image_cache[image_hash]
            # Still update environment tracking with cached description
            detected_env = self.detect_environment(cached_description)
            self.update_environment_tracking(detected_env)
            utils.log(f"Using cached frame description (environment: {self.current_environment})")
            return cached_description
            
        # Check if this is the same as last frame
        if image_hash == self.last_frame_hash:
            cached_description = self.image_cache.get(image_hash, "Game screen with no changes")
            utils.log("Frame unchanged, reusing last description")
            return cached_description
        
        self.last_frame_hash = image_hash
        
        # Choose prompt based on recent environment history for better specificity
        prompt_to_use = self.base_prompt
        
        # If we've consistently detected a cave, use specialized cave prompt
        if any(env in ["CAVE", "MT_MOON", "ROCK_TUNNEL", "VICTORY_ROAD"] for env in self.environment_history[-2:]):
            prompt_to_use = self.specialized_prompts["CAVE_DETECTION"]
            utils.log("Using specialized cave detection prompt")
        
        # If we've detected a dark cave, use specialized dark cave prompt
        elif "DARK_CAVE" in self.environment_history[-2:]:
            prompt_to_use = self.specialized_prompts["DARK_CAVE_DETECTION"]
            utils.log("Using specialized dark cave detection prompt")
        
        # Generate caption using Moondream
        with torch.inference_mode():
            caption = self.model.query(
                image,
                prompt_to_use,
            )["answer"]
            
        utils.log(f"Moondream Screen Description: {caption}")
        
        # Update environment tracking
        detected_env = self.detect_environment(caption)
        self.update_environment_tracking(detected_env)
        
        # Update cache
        if len(self.image_cache) >= self.cache_limit:
            oldest = next(iter(self.image_cache))
            del self.image_cache[oldest]
        self.image_cache[image_hash] = caption
        
        # More precise title screen tracking
        if ("title screen" in caption.lower() or "blue version" in caption.lower()) and \
           ("professor" not in caption.lower() and "oak" not in caption.lower()):
            self.consecutive_title_screens += 1
            utils.log(f"Title screen detection count: {self.consecutive_title_screens}")
        else:
            self.consecutive_title_screens = 0
            
        return caption

    def get_context_aware_prompt(self, task_instruction=None):
        """
        Generate a context-aware prompt based on the current environment.
        """
        # Base prompt parts
        base_prefix = "You are playing Pokémon Blue. "
        base_suffix = "\nReply with ONLY one button: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT"
        
        # Environment-specific instructions
        env_instructions = {
            "DARK_CAVE": "You're in a dark cave with limited visibility. Navigate carefully and consider using Flash. ",
            "CAVE": "You're exploring a cave. Watch for wild Pokémon and look for items. ",
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
        
        # Get environment-specific instruction or use general default
        env_instruction = env_instructions.get(self.current_environment, "Choose a button based on what you see. ")
        
        # Include task instruction if provided, otherwise use environment-based guidance
        if task_instruction:
            prompt = f"{base_prefix}Choose a button based on this instruction:\n{task_instruction}{base_suffix}"
        else:
            prompt = f"{base_prefix}{env_instruction}{base_suffix}"
            
        return prompt

    def get_next_action(self, task_instruction=None):
        """
        Use the Moondream model to choose the next button press with environment awareness.
        """
        # Title screen fast path
        if self.consecutive_title_screens >= 2:
            utils.log("Multiple title screens detected. Forcing START button.")
            return "START"
        
        image = self.capture_screen()
        
        # Generate context-aware prompt
        prompt = self.get_context_aware_prompt(task_instruction)
        utils.log(f"Using prompt: {prompt}")
        
        # Generate with optimized settings
        with torch.inference_mode():
            response = self.model.query(
                image,
                prompt,
            )["answer"]
            
        utils.log(f"Moondream raw decision: {response}")
        action = response.strip().upper()
        
        # More specific title screen check to avoid false triggers
        if ("TITLE SCREEN" in action or "BLUE VERSION" in action) and \
           ("PROFESSOR" not in action and "OAK" not in action):
            utils.log("Title screen text detected in response, overriding to START")
            return "START"
        
        # Extract button by direct pattern matching
        for button in self.valid_buttons:
            if button in action:
                utils.log(f"Extracted button '{button}' from '{action}'")
                return button
                
        # Environment-specific fallbacks
        if self.current_environment == "DARK_CAVE":
            utils.log("In dark cave with no clear action. Defaulting to START to access Flash.")
            return "START"
        
        if self.current_environment == "DIALOGUE":
            utils.log("In dialogue with no clear action. Defaulting to A.")
            return "A"
        
        # Default to "redo" if no valid button found - NEEDED FOR AI PLAYTHROUGH
        utils.log(f"No valid button in '{action}'. Defaulting to RETRYING.")
        return self.get_next_action()  # Recursive call to try again