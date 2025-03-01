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
    Decision layer for Pokémon Blue that uses the Llama-3.2-3B model to generate a single-button instruction.
    The prompt templates force the model to respond with exactly one button from the allowed list:
    A, B, START, SELECT, UP, DOWN, LEFT, RIGHT.
    This module leverages Vision’s OCR, environment detection, and game state (inventory/party)
    to decide the next button to press for navigating the menus or overworld.
    Ghost encounters override the state to "GHOST" so that the prompt instructs using the Sylph Scope
    (if obtained) or running from battle otherwise.
    """
    ALLOWED_BUTTONS = {"A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT"}

    def __init__(self, vision, device=None):
        self.vision = vision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.llama_model_name = "meta-llama/Llama-3.2-3B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.llama_model_name)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False
        )
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            self.llama_model_name,
            device_map="cuda",
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        if torch.cuda.is_available() and config.TORCH_COMPILE:
            self.llama_model = torch.compile(self.llama_model)
        stop_words = [".", "\n"]
        stop_token_ids = [self.tokenizer.encode(word, add_special_tokens=False)[-1] for word in stop_words]
        self.stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
        utils.log(f"Llama model initialized on: {self.device} with 8-bit quantization (no CPU offload) and float16 precision")
        self.prompt_templates = self._create_prompt_templates()
        self.current_state = "UNKNOWN"
        self.previous_states = []
        self.state_transitions = 0
        self.instruction_cache = {}
        self.cache_size = 20
        self.common_state_actions = {
            "TITLE SCREEN": "START",
            "DIALOGUE": "A",
            "INTRO_SEQUENCE": "A",
            "ITEM_FOUND": "A",
            "HEALING": "A",
            "EVOLUTION": "A"
        }
        self.pokemon_caught = False
        self.sylph_scope_obtained = False

    def _create_prompt_templates(self):
        # Each template now instructs the model to reply with exactly one allowed button.
        base_line = "Reply with exactly one button from: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT."
        return {
            "TITLE SCREEN": "You are at the Pokémon Blue title screen. Navigate to 'NEW GAME'. " + base_line,
            "INTRO_SEQUENCE": "You are in Professor Oak's introduction. Continue the dialogue. " + base_line,
            "DIALOGUE": "You are in a dialogue screen. Choose the next option. " + base_line,
            "MENU": "You are in a menu. Navigate the selection. " + base_line,
            "OVERWORLD": "You are exploring the overworld. Choose your direction or interaction. " + base_line,
            "BATTLE": "You are in a battle menu. Choose your move. " + base_line,
            "WILD_BATTLE": "You are in a wild battle. Choose your move option. " + base_line,
            "TRAINER_BATTLE": "You are in a trainer battle. Select your move. " + base_line,
            "GYM_LEADER_BATTLE": "You are battling a Gym Leader. Select the best move. " + base_line,
            "CAVE": "You are in a cave. Choose a direction to explore. " + base_line,
            "DARK_CAVE": "You are in a dark cave. Choose a direction cautiously; if Flash is available, select it. " + base_line,
            "MT MOON": "You are in Mt. Moon. Choose your navigation option. " + base_line,
            "ROCK_TUNNEL": "You are in Rock Tunnel. Choose a direction to bypass obstacles. " + base_line,
            "VICTORY ROAD": "You are in Victory Road. Choose your next move carefully. " + base_line,
            "BUILDING": "You are inside a building. Choose your interaction. " + base_line,
            "POKEMON CENTER": "You are in a Pokémon Center. Choose your menu option to heal or talk. " + base_line,
            "POKEMON_MART": "You are in a Pokémon Mart. Choose your menu option to browse items. " + base_line,
            "GYM": "You are in a Pokémon Gym. Choose your navigation option. " + base_line,
            "ITEM_FOUND": "An item is present. Choose the option to pick it up. " + base_line,
            "HEALING": "Your Pokémon are being healed. Choose to proceed after healing. " + base_line,
            "EVOLUTION": "Your Pokémon is evolving. Choose to confirm or cancel the evolution. " + base_line,
            "FISHING": "You are fishing. Choose to reel in your catch. " + base_line,
            "POKEMON_MENU": "You are in the Pokémon party screen. Choose a Pokémon to select. " + base_line,
            "ITEM_MENU": "You are in the item menu. Choose an item to use. " + base_line,
            "SAVE_MENU": "You are at the save prompt. Choose to save or cancel. " + base_line,
            "STRENGTH_PUZZLE": "You are at a strength puzzle. Choose your action to solve it. " + base_line,
            "CUT_OBSTACLE": "A tree is blocking your path. Choose the action to use Cut. " + base_line,
            "SURF_WATER": "You are at a body of water. Choose the action to use Surf. " + base_line,
            "GHOST": "You are encountering a ghost in the cemetery. If you have the Sylph Scope, choose the action to use it; otherwise, choose to run. " + base_line
        }

    def detect_game_state(self, game_state_text):
        game_state_upper = game_state_text.upper()
        if ("PROFESSOR" in game_state_upper or "OAK" in game_state_upper) and "INTRO" in game_state_upper:
            return "INTRO_SEQUENCE"
        if "BLUE VERSION" in game_state_upper or "TITLE SCREEN" in game_state_upper:
            return "TITLE SCREEN"
        if "GYM LEADER" in game_state_upper and "BATTLE" in game_state_upper:
            return "GYM_LEADER_BATTLE"
        if "TRAINER" in game_state_upper and "BATTLE" in game_state_upper:
            return "TRAINER_BATTLE"
        if "WILD" in game_state_upper and "BATTLE" in game_state_upper:
            return "WILD_BATTLE"
        if "BATTLE" in game_state_upper or "FIGHT" in game_state_upper:
            return "BATTLE"
        if ("DARK" in game_state_upper or "CAN'T SEE" in game_state_upper) and ("CAVE" in game_state_upper or "TUNNEL" in game_state_upper):
            return "DARK_CAVE"
        if "MT MOON" in game_state_upper or "MT. MOON" in game_state_upper:
            return "MT MOON"
        if "ROCK TUNNEL" in game_state_upper:
            return "ROCK_TUNNEL"
        if "VICTORY ROAD" in game_state_upper:
            return "VICTORY ROAD"
        if "CAVE" in game_state_upper:
            return "CAVE"
        if "POKEMON CENTER" in game_state_upper or "POKÉMON CENTER" in game_state_upper:
            return "POKEMON CENTER"
        if "MART" in game_state_upper or "SHOP" in game_state_upper or "STORE" in game_state_upper:
            return "POKEMON_MART"
        if "GYM" in game_state_upper:
            return "GYM"
        if "BUILDING" in game_state_upper or "HOUSE" in game_state_upper or "INDOOR" in game_state_upper:
            return "BUILDING"
        if "FOUND" in game_state_upper and "ITEM" in game_state_upper:
            return "ITEM_FOUND"
        if "HEALING" in game_state_upper or "NURSE JOY" in game_state_upper:
            return "HEALING"
        if "EVOLVING" in game_state_upper or "EVOLUTION" in game_state_upper:
            return "EVOLUTION"
        if "FISHING" in game_state_upper or "FISHING ROD" in game_state_upper:
            return "FISHING"
        if "POKEMON" in game_state_upper and "MENU" in game_state_upper:
            return "POKEMON_MENU"
        if "ITEM" in game_state_upper and "MENU" in game_state_upper:
            return "ITEM_MENU"
        if "SAVE" in game_state_upper:
            return "SAVE_MENU"
        if "MENU" in game_state_upper or "OPTION" in game_state_upper:
            return "MENU"
        if "BOULDER" in game_state_upper and "STRENGTH" in game_state_upper:
            return "STRENGTH_PUZZLE"
        if "TREE" in game_state_upper and "CUT" in game_state_upper:
            return "CUT_OBSTACLE"
        if ("WATER" in game_state_upper or "LAKE" in game_state_upper or "OCEAN" in game_state_upper) and "SURF" in game_state_upper:
            return "SURF_WATER"
        if "TEXT" in game_state_upper or "DIALOGUE" in game_state_upper:
            return "DIALOGUE"
        return "OVERWORLD"

    def get_llama_instruction(self, game_state):
        if hasattr(self.vision, 'current_environment') and self.vision.current_environment != "UNKNOWN":
            if self.vision.environment_confidence > 0.7:
                new_state = self.vision.current_environment
                utils.log(f"Using Vision's environment detection: {new_state} (confidence: {self.vision.environment_confidence:.2f})")
            else:
                new_state = self.detect_game_state(game_state)
        else:
            new_state = self.detect_game_state(game_state)
        if self.vision.found_ghost:
            new_state = "GHOST"
            utils.log("Overriding state to GHOST due to ghost detection.")
        if new_state != self.current_state:
            self.previous_states.append(self.current_state)
            if len(self.previous_states) > 5:
                self.previous_states.pop(0)
            self.state_transitions += 1
            utils.log(f"State transition: {self.current_state} -> {new_state}")
            self.current_state = new_state
        if new_state in self.common_state_actions:
            # For common states, return the corresponding button.
            return self.common_state_actions[new_state]
        cache_key = f"{new_state}:{hash(game_state[:100])}"
        if cache_key in self.instruction_cache:
            cached_instruction = self.instruction_cache[cache_key]
            utils.log(f"Using cached instruction for {new_state}: {cached_instruction}")
            return cached_instruction
        prompt = self.prompt_templates.get(new_state, self.prompt_templates["OVERWORLD"])
        utils.log(f"Generating instruction for state: {new_state}")
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device).to(torch.long)
        with torch.inference_mode():
            temperature = 0.5
            if new_state in ["DARK_CAVE", "MT MOON", "ROCK_TUNNEL", "VICTORY ROAD"]:
                temperature = 0.3
            elif new_state in ["BATTLE", "WILD_BATTLE", "TRAINER_BATTLE", "GYM_LEADER_BATTLE"]:
                temperature = 0.4
            output = self.llama_model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=temperature,
                top_p=0.85,
                repetition_penalty=1.1,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id
            )
        instruction = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
        utils.log(f"Llama raw instruction: {instruction}")
        # Ensure the output is one of the allowed buttons.
        if instruction not in self.ALLOWED_BUTTONS:
            utils.log(f"Instruction '{instruction}' is invalid. Defaulting to 'A'.")
            instruction = "A"
        if len(self.instruction_cache) >= self.cache_size:
            oldest_key = next(iter(self.instruction_cache))
            del self.instruction_cache[oldest_key]
        self.instruction_cache[cache_key] = instruction
        return instruction

    def handle_dark_cave(self):
        if len(self.previous_states) >= 3 and all(state == "DARK_CAVE" for state in self.previous_states[-3:]):
            utils.log("Stuck in dark cave, attempting to access Flash via menu.")
            return "START"
        last_actions = getattr(self, '_last_dark_cave_actions', [])
        if not last_actions:
            utils.log("Starting dark cave navigation with 'RIGHT'.")
            self._last_dark_cave_actions = ["RIGHT"]
            return "RIGHT"
        last_action = last_actions[-1]
        if len(last_actions) >= 2 and last_actions[-1] == last_actions[-2]:
            directions = ["UP", "RIGHT", "DOWN", "LEFT"]
            current_idx = directions.index(last_action)
            next_idx = (current_idx + 1) % 4
            next_action = directions[next_idx]
            utils.log(f"Changing direction in dark cave from {last_action} to {next_action}.")
            last_actions.append(next_action)
            self._last_dark_cave_actions = last_actions[-5:] if len(last_actions) > 5 else last_actions
            return next_action
        else:
            last_actions.append(last_action)
            self._last_dark_cave_actions = last_actions[-5:] if len(last_actions) > 5 else last_actions
            return last_action

    def get_next_action(self):
        game_state = self.vision.get_game_state_text()
        utils.log(f"Game state from vision: {game_state}")
        if hasattr(self.vision, 'found_pokemon') and self.vision.found_pokemon:
            self.pokemon_caught = True
            utils.log("Detected Pokémon via OCR: " + ", ".join(self.vision.found_pokemon))
        if self.vision.found_ghost and not self.sylph_scope_obtained:
            utils.log("Ghost detected and Sylph Scope not obtained. Instructing to run from battle.")
            return "B"  # Assume B is used to run (or map accordingly)
        elif self.vision.found_ghost and self.sylph_scope_obtained:
            utils.log("Ghost detected and Sylph Scope is available. Proceed with ghost instructions.")
        instruction_text = self.get_llama_instruction(game_state)
        utils.log(f"Final button decision: {instruction_text}")
        return instruction_text
