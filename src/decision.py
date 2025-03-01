import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
import utils
import config

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids, scores, **kwargs):
        return any(input_ids[0][-1] == stop_id for stop_id in self.stop_token_ids)

class Decision:
    """
    Decision layer that leverages Llama-3.2-3B to generate concise task instructions
    from the game state and then delegates to Moondream (via Vision) to determine the
    next button press.
    """
    def __init__(self, vision, device=None):
        self.vision = vision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.llama_model_name = "meta-llama/Llama-3.2-3B"
        
        # Initialize Llama tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.llama_model_name)
        
        # Use BitsAndBytesConfig for proper 8-bit quantization settings
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False  # Ensure entire model stays in VRAM
        )
        
        # Load Llama model with 8-bit quantization and float16 precision
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            self.llama_model_name,
            device_map="cuda",  # Force all modules on GPU
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Optional: Compile the model for faster inference (if using CUDA)
        if torch.cuda.is_available() and config.TORCH_COMPILE:
            self.llama_model = torch.compile(self.llama_model)
        
        # Stop tokens for controlled generation
        stop_words = [".", "\n"]
        stop_token_ids = [self.tokenizer.encode(word, add_special_tokens=False)[-1] for word in stop_words]
        self.stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
        
        utils.log(f"Llama model initialized on: {self.device} with 8-bit quantization (no CPU offload) and float16 precision")
        
        # Precompile prompts for faster generation
        self.prompt_templates = self._create_prompt_templates()
        
        # Track the current game state
        self.current_state = "UNKNOWN"
        self.previous_states = []
        self.state_transitions = 0
        
        # Improved cache for recently seen game states
        self.instruction_cache = {}
        self.cache_size = 20  

        # Direct action mapping for common states
        self.common_state_actions = {
            "TITLE_SCREEN": "START",
            "DIALOGUE": "A",
            "INTRO_SEQUENCE": "A",
            "ITEM_FOUND": "A",
            "HEALING": "A",
            "EVOLUTION": "A"
        }

    def _create_prompt_templates(self):
        """Create optimized prompt templates for Llama"""
        return {
            # General game states
            "TITLE_SCREEN": "You're at the Pokémon Blue title screen. Respond with: 'Press START button to begin the game.'",
            "INTRO_SEQUENCE": "You're in Professor Oak's introduction. Respond with: 'Press A button to continue dialogue.'",
            "DIALOGUE": "You're in a dialogue screen. Respond with: 'Press A button to continue dialogue.'",
            "MENU": "You're navigating a menu. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Press A', 'Press B'.",
            "OVERWORLD": "You're exploring the overworld. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Move LEFT', 'Move RIGHT', 'Press A'.",
            
            # Battle states with specificity
            "BATTLE": "You're in a Pokémon battle. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Move LEFT', 'Move RIGHT', 'Press A', 'Press B'.",
            "WILD_BATTLE": "You're in a wild Pokémon battle. Choose 'FIGHT' for moves, or other options. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Move LEFT', 'Move RIGHT', 'Press A', 'Press B'.",
            "TRAINER_BATTLE": "You're in a trainer battle. Choose your moves wisely. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Move LEFT', 'Move RIGHT', 'Press A', 'Press B'.",
            "GYM_LEADER_BATTLE": "You're battling a Gym Leader. This is an important battle for a badge. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Move LEFT', 'Move RIGHT', 'Press A', 'Press B'.",
            
            # Cave-specific templates
            "CAVE": "You're exploring a cave in Pokémon Blue. Watch for wild Pokémon encounters and look for items. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Move LEFT', 'Move RIGHT', 'Press A', 'Press START' (to access menu).",
            "DARK_CAVE": "You're in a dark cave without Flash. You can only see a small area around your character. Navigate carefully or use Flash. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Move LEFT', 'Move RIGHT', 'Press START' (to use Flash if available).",
            "MT_MOON": "You're in Mt. Moon. Watch for Zubat encounters and look for items and fossil researchers. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Move LEFT', 'Move RIGHT', 'Press A', 'Press START'.",
            "ROCK_TUNNEL": "You're in Rock Tunnel. This area is complex with many trainers. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Move LEFT', 'Move RIGHT', 'Press A', 'Press START'.",
            "VICTORY_ROAD": "You're in Victory Road. This is a challenging cave with strength puzzles and strong trainers. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Move LEFT', 'Move RIGHT', 'Press A', 'Press START'.",
            
            # Location-specific templates
            "BUILDING": "You're inside a building. Look for NPCs to talk to or items. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Move LEFT', 'Move RIGHT', 'Press A'.",
            "POKEMON_CENTER": "You're in a Pokémon Center. Talk to Nurse Joy to heal or use the PC. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Move LEFT', 'Move RIGHT', 'Press A'.",
            "POKEMON_MART": "You're in a Pokémon Mart. Talk to the clerk to buy items. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Move LEFT', 'Move RIGHT', 'Press A'.",
            "GYM": "You're in a Pokémon Gym. Navigate to the Gym Leader while defeating trainers. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Move LEFT', 'Move RIGHT', 'Press A'.",
            
            # Special interaction templates
            "ITEM_FOUND": "You've found an item. Respond with: 'Press A to pick up the item.'",
            "HEALING": "Your Pokémon are being healed. Respond with: 'Press A to continue.'",
            "EVOLUTION": "Your Pokémon is evolving! Respond with: 'Press A to continue evolution' or 'Press B to cancel evolution'.",
            "FISHING": "You're using a fishing rod. Wait for a bite. Respond with: 'Press A when there's a bite.'",
            
            # Menu-specific templates
            "POKEMON_MENU": "You're in the Pokémon menu. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Press A to select', 'Press B to exit'.",
            "ITEM_MENU": "You're in the Item menu. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Press A to use item', 'Press B to exit'.",
            "SAVE_MENU": "You're at the save prompt. Respond with: 'Press A to save' or 'Press B to cancel'.",
            
            # Puzzle-specific templates
            "STRENGTH_PUZZLE": "You're at a strength puzzle. You need to push boulders. Respond with: 'Press A to use Strength' or 'Move in the direction to push boulder'.",
            "CUT_OBSTACLE": "There's a small tree blocking your path. Respond with: 'Press A to use Cut' if facing the tree.",
            "SURF_WATER": "You're at a body of water. Respond with: 'Press A to use Surf' if facing the water."
        }

    def detect_game_state(self, game_state_text):
        """
        Analyzes the game state description to determine the player's situation.
        """
        game_state_upper = game_state_text.upper()

        # More precise detection logic with priority order
        if ("PROFESSOR" in game_state_upper or "OAK" in game_state_upper) and "INTRO" in game_state_upper:
            return "INTRO_SEQUENCE"
        if "BLUE VERSION" in game_state_upper or "TITLE SCREEN" in game_state_upper:
            return "TITLE_SCREEN"
            
        # Battle detection with specificity
        if "GYM LEADER" in game_state_upper and "BATTLE" in game_state_upper:
            return "GYM_LEADER_BATTLE"
        if "TRAINER" in game_state_upper and "BATTLE" in game_state_upper:
            return "TRAINER_BATTLE"
        if "WILD" in game_state_upper and "BATTLE" in game_state_upper:
            return "WILD_BATTLE"
        if "BATTLE" in game_state_upper or "FIGHT" in game_state_upper:
            return "BATTLE"
            
        # Cave detection with specificity
        if ("DARK" in game_state_upper or "CAN'T SEE" in game_state_upper) and ("CAVE" in game_state_upper or "TUNNEL" in game_state_upper):
            return "DARK_CAVE"
        if "MT MOON" in game_state_upper or "MT. MOON" in game_state_upper:
            return "MT_MOON"
        if "ROCK TUNNEL" in game_state_upper:
            return "ROCK_TUNNEL"
        if "VICTORY ROAD" in game_state_upper:
            return "VICTORY_ROAD"
        if "CAVE" in game_state_upper:
            return "CAVE"
            
        # Building detection
        if "POKEMON CENTER" in game_state_upper or "POKÉMON CENTER" in game_state_upper:
            return "POKEMON_CENTER"
        if "MART" in game_state_upper or "SHOP" in game_state_upper or "STORE" in game_state_upper:
            return "POKEMON_MART"
        if "GYM" in game_state_upper:
            return "GYM"
        if "BUILDING" in game_state_upper or "HOUSE" in game_state_upper or "INDOOR" in game_state_upper:
            return "BUILDING"
            
        # Special interactions
        if "FOUND" in game_state_upper and "ITEM" in game_state_upper:
            return "ITEM_FOUND"
        if "HEALING" in game_state_upper or "NURSE JOY" in game_state_upper:
            return "HEALING"
        if "EVOLVING" in game_state_upper or "EVOLUTION" in game_state_upper:
            return "EVOLUTION"
        if "FISHING" in game_state_upper or "FISHING ROD" in game_state_upper:
            return "FISHING"
            
        # Menu detection
        if "POKEMON" in game_state_upper and "MENU" in game_state_upper:
            return "POKEMON_MENU"
        if "ITEM" in game_state_upper and "MENU" in game_state_upper:
            return "ITEM_MENU"
        if "SAVE" in game_state_upper:
            return "SAVE_MENU"
        if "MENU" in game_state_upper or "OPTION" in game_state_upper:
            return "MENU"
            
        # Puzzle detection
        if "BOULDER" in game_state_upper and "STRENGTH" in game_state_upper:
            return "STRENGTH_PUZZLE"
        if "TREE" in game_state_upper and "CUT" in game_state_upper:
            return "CUT_OBSTACLE"
        if ("WATER" in game_state_upper or "LAKE" in game_state_upper or "OCEAN" in game_state_upper) and "SURF" in game_state_upper:
            return "SURF_WATER"
            
        # Text/dialogue detection
        if "TEXT" in game_state_upper or "DIALOGUE" in game_state_upper:
            return "DIALOGUE"
        
        # Default to overworld if no specific state is detected
        return "OVERWORLD"

    def get_llama_instruction(self, game_state):
        """
        Uses Llama-3.2-3B to generate a concise instruction based on the current game state.
        """
        # First check if we can use the vision's environment detection
        if hasattr(self.vision, 'current_environment') and self.vision.current_environment != "UNKNOWN":
            # Use the environment detection from Vision if available and confident
            if self.vision.environment_confidence > 0.7:  # Only use if confidence is high
                new_state = self.vision.current_environment
                utils.log(f"Using Vision's environment detection: {new_state} (confidence: {self.vision.environment_confidence:.2f})")
            else:
                # Fall back to text-based detection if confidence is low
                new_state = self.detect_game_state(game_state)
        else:
            # Fall back to traditional detection if Vision doesn't provide environment info
            new_state = self.detect_game_state(game_state)
        
        # Track state transitions
        if new_state != self.current_state:
            self.previous_states.append(self.current_state)
            if len(self.previous_states) > 5:  # Keep only the last 5 states
                self.previous_states.pop(0)
                
            self.state_transitions += 1
            utils.log(f"State transition: {self.current_state} -> {new_state}")
            self.current_state = new_state

        # Direct action mapping for common states
        if new_state in self.common_state_actions:
            return f"Press {self.common_state_actions[new_state]} button."

        # Intelligent caching: Check if we've seen this state recently
        cache_key = f"{new_state}:{hash(game_state[:100])}"  # Use first 100 chars for hashing
        if cache_key in self.instruction_cache:
            cached_instruction = self.instruction_cache[cache_key]
            utils.log(f"Using cached instruction for {new_state}: {cached_instruction}")
            return cached_instruction

        # Use state-based prompt template
        prompt = self.prompt_templates.get(new_state, self.prompt_templates["OVERWORLD"])
        utils.log(f"Generating instruction for state: {new_state}")

        # Tokenize and ensure input is float16
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device).to(torch.long)

        # Generate response with optimized parameters for each state type
        with torch.inference_mode():
            # Adjust generation parameters based on state type
            temperature = 0.5  # Default temperature
            if new_state in ["DARK_CAVE", "MT_MOON", "ROCK_TUNNEL", "VICTORY_ROAD"]:
                temperature = 0.3  # More deterministic for caves
            elif new_state in ["BATTLE", "WILD_BATTLE", "TRAINER_BATTLE", "GYM_LEADER_BATTLE"]:
                temperature = 0.4  # Slightly more deterministic for battles
                
            output = self.llama_model.generate(
                input_ids, 
                max_new_tokens=25,
                temperature=temperature,
                top_p=0.85,
                repetition_penalty=1.1,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode output
        instruction = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        utils.log(f"Llama instruction: {instruction}")

        # Cache the instruction
        if len(self.instruction_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.instruction_cache))
            del self.instruction_cache[oldest_key]
        self.instruction_cache[cache_key] = instruction

        return instruction if instruction else "Press A button to continue."
    
    def handle_dark_cave(self):
        """Special handling for dark caves without Flash"""
        # Check if we've been in dark cave for multiple consecutive states
        if len(self.previous_states) >= 3 and all(state == "DARK_CAVE" for state in self.previous_states[-3:]):
            utils.log("Stuck in dark cave, attempting to access Flash through menu")
            return "START"  # Access menu to potentially use Flash
        
        # Implement a simple wall-following algorithm for dark caves
        last_actions = getattr(self, '_last_dark_cave_actions', [])
        if not last_actions:
            utils.log("Starting dark cave navigation with RIGHT direction")
            self._last_dark_cave_actions = ["RIGHT"]
            return "RIGHT"
            
        # Simple navigation pattern: try to follow the right wall
        last_action = last_actions[-1]
        if len(last_actions) >= 2 and last_actions[-1] == last_actions[-2]:
            # If we've tried the same direction twice, try a different direction
            directions = ["UP", "RIGHT", "DOWN", "LEFT"]
            current_idx = directions.index(last_action)
            next_idx = (current_idx + 1) % 4
            next_action = directions[next_idx]
            utils.log(f"Changing direction in dark cave from {last_action} to {next_action}")
            last_actions.append(next_action)
            self._last_dark_cave_actions = last_actions[-5:] if len(last_actions) > 5 else last_actions
            return next_action
        else:
            # Continue in the same direction
            last_actions.append(last_action)
            self._last_dark_cave_actions = last_actions[-5:] if len(last_actions) > 5 else last_actions
            return last_action

    def get_next_action(self):
        """
        Uses Vision to obtain the game state and determine the next button press.
        """
        game_state = self.vision.get_game_state_text()
        utils.log(f"Game state from vision: {game_state}")

        # Special handling for common states
        if "TITLE SCREEN" in game_state.upper() or self.current_state == "TITLE_SCREEN":
            return "START"
            
        # Special handling for dark caves
        if self.current_state == "DARK_CAVE":
            return self.handle_dark_cave()

        # Generate task instruction
        instruction = self.get_llama_instruction(game_state)
        utils.log(f"Llama instruction: {instruction}")

        # Pass instruction to Vision for context-aware decision
        if hasattr(self.vision, 'get_next_action') and callable(getattr(self.vision, 'get_next_action')):
            action = self.vision.get_next_action(instruction)
            utils.log(f"Vision-based action with instruction: {action}")
            return action

        # Fallback extraction from instruction string if Vision doesn't provide action
        action = instruction.upper()
        if "PRESS A" in action:
            return "A"
        if "PRESS B" in action:
            return "B"
        if "PRESS START" in action:
            return "START"
        if "MOVE UP" in action:
            return "UP"
        if "MOVE DOWN" in action:
            return "DOWN"
        if "MOVE LEFT" in action:
            return "LEFT"
        if "MOVE RIGHT" in action:
            return "RIGHT"
        
        # Recursive call to try again for AI PLAYTHROUGH
        return self.get_next_action()  # Recursive call NEEDED AND NOT TO BE REMOVED for AI PLAYTHROUGH