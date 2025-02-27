import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import utils

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
        
        # Initialize Llama tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(self.llama_model_name)
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            self.llama_model_name,
            device_map={"": self.device}
        )
        utils.log(f"Llama model initialized on: {self.device}")
        
        # Track the current game state
        self.current_state = "UNKNOWN"
        self.previous_states = []
        self.state_transitions = 0

    def detect_game_state(self, game_state_text):
        """
        Analyzes the game state description to determine what stage of the game we're in.
        Returns a string representing the current game state.
        """
        if self.current_state != "UNKNOWN":
            self.previous_states.append(self.current_state)
            if len(self.previous_states) > 5:
                self.previous_states.pop(0)
        
        if ("BLUE VERSION" in game_state_text.upper() or "POKÉMON LOGO" in game_state_text.upper()) and "OAK" not in game_state_text.upper():
            return "TITLE_SCREEN"
        elif "OAK" in game_state_text.upper() or "PROFESSOR" in game_state_text.upper():
            return "INTRO_SEQUENCE"
        elif "TEXT" in game_state_text.upper() or "DIALOGUE" in game_state_text.upper():
            return "DIALOGUE"
        elif "BATTLE" in game_state_text.upper() or "FIGHT" in game_state_text.upper() or "ATTACK" in game_state_text.upper():
            return "BATTLE"
        elif "MENU" in game_state_text.upper() or "OPTION" in game_state_text.upper() or "ITEMS" in game_state_text.upper():
            return "MENU"
        else:
            return "OVERWORLD"

    def get_llama_instruction(self, game_state):
        """
        Uses Llama to convert a game state description into a concise instruction task.
        Instead of giving up after a fixed number of attempts, this method will
        continuously modify generation settings until a valid instruction is produced.
        """
        new_state = self.detect_game_state(game_state)
        if new_state != self.current_state:
            self.state_transitions += 1
            utils.log(f"State transition: {self.current_state} -> {new_state}")
            self.current_state = new_state

        # Build a state-specific prompt.
        if self.current_state == "TITLE_SCREEN":
            prompt = f"""
You are an expert Pokémon Blue player. You're looking at the title screen now.

Game state description:
\"\"\"{game_state}\"\"\"

The correct action on the title screen is to press START (not A).
Your response must be EXACTLY "Press START button to begin the game."

Don't output any other buttons or explanations.
"""
        elif self.current_state == "INTRO_SEQUENCE":
            prompt = f"""
You are an expert Pokémon Blue player. You're in the introduction sequence with Professor Oak.

Game state description:
\"\"\"{game_state}\"\"\"

To progress through the introduction, we should press A to continue the dialogue.
Your response must be EXACTLY "Press A button to continue dialogue."

Don't output any other buttons or explanations.
"""
        elif self.current_state == "DIALOGUE":
            prompt = f"""
You are an expert Pokémon Blue player. You're in dialogue with an NPC or reading text.

Game state description:
\"\"\"{game_state}\"\"\"

To continue reading text or progress the dialogue, press A.
Your response must be EXACTLY "Press A button to continue dialogue."

Don't output any other buttons or explanations.
"""
        elif self.current_state == "BATTLE":
            prompt = f"""
You are an expert Pokémon Blue player. You're in a Pokémon battle.

Game state description:
\"\"\"{game_state}\"\"\"

Based on the battle screen, determine if you need to select a move (UP/DOWN then A) or perform another action.
Respond with EXACTLY one of these formats:
"Move UP to navigate battle menu."
"Move DOWN to navigate battle menu."
"Press A to select the current option."

Don't output any other buttons or explanations.
"""
        elif self.current_state == "MENU":
            prompt = f"""
You are an expert Pokémon Blue player. You're navigating a menu.

Game state description:
\"\"\"{game_state}\"\"\"

Based on the menu screen, determine how to navigate:
Respond with EXACTLY one of these formats:
"Move UP to navigate menu."
"Move DOWN to navigate menu."
"Press A to select the current option."
"Press B to exit the menu."

Don't output any other buttons or explanations.
"""
        else:  # OVERWORLD or UNKNOWN
            prompt = f"""
You are an expert Pokémon Blue player. You're exploring the overworld.

Game state description:
\"\"\"{game_state}\"\"\"

Based on the screen, determine which direction to move or button to press:
Respond with EXACTLY one of these formats:
"Move UP to explore."
"Move DOWN to explore."
"Move LEFT to explore."
"Move RIGHT to explore."
"Press A to interact."
"Press START to open menu."

Don't output any other buttons or explanations.
"""

        utils.log(f"Generating instruction for state: {self.current_state}")
        
        attempt = 0
        base_temperature = 0.7
        base_max_tokens = 30
        valid_indicators = ["PRESS A", "PRESS B", "PRESS START", "MOVE UP", "MOVE DOWN", "MOVE LEFT", "MOVE RIGHT"]
        while True:
            curr_temp = base_temperature + (0.1 * (attempt % 10))
            curr_max_tokens = base_max_tokens + (2 * (attempt // 10))
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            output_ids = self.llama_model.generate(input_ids, max_new_tokens=curr_max_tokens, temperature=curr_temp)
            instruction = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            instruction = instruction.replace(prompt, "").strip()
            utils.log(f"Attempt {attempt+1} (temp={curr_temp}, max_tokens={curr_max_tokens}): Instruction: {instruction}")
            if any(ind in instruction.upper() for ind in valid_indicators):
                return instruction.strip()
            attempt += 1

    def get_next_action(self):
        """
        Uses Vision to obtain the game state text, then returns a button press decision.
        If the game state indicates the title screen, it immediately forces the START button,
        bypassing instruction generation.
        Otherwise, it continuously adjusts generation settings until a valid instruction
        is parsed.
        """
        game_state = self.vision.get_game_state_text()
        utils.log(f"Game state from vision: {game_state}")
        
        # Immediately force START if title screen is detected.
        if "BLUE VERSION" in game_state.upper() or "POKÉMON LOGO" in game_state.upper():
            utils.log("Title screen detected from game state. Forcing START button.")
            self.current_state = "TITLE_SCREEN"
            return "START"

        attempt = 0
        while True:
            try:
                instruction = self.get_llama_instruction(game_state)
            except Exception as e:
                utils.log(f"Attempt {attempt+1}: Llama generation failed: {e}")
                continue

            utils.log(f"Llama instruction: {instruction}")
            if self.current_state == "TITLE_SCREEN":
                utils.log("Title screen detected. Forcing START button.")
                return "START"

            instruction_upper = instruction.upper()
            if "PRESS A" in instruction_upper:
                return "A"
            elif "PRESS B" in instruction_upper:
                return "B"
            elif "PRESS START" in instruction_upper:
                return "START"
            elif "MOVE UP" in instruction_upper:
                return "UP"
            elif "MOVE DOWN" in instruction_upper:
                return "DOWN"
            elif "MOVE LEFT" in instruction_upper:
                return "LEFT"
            elif "MOVE RIGHT" in instruction_upper:
                return "RIGHT"
            else:
                utils.log(f"Attempt {attempt+1}: No valid action parsed from instruction: '{instruction}'. Retrying with new generation settings...")
            attempt += 1
