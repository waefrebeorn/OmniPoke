import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import utils

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
        
        # Load Llama with full float16 precision
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            self.llama_model_name,
            device_map="auto",  # ✅ Auto-selects CUDA or CPU
            torch_dtype=torch.float16,  # ✅ Full FP16 Mode
            low_cpu_mem_usage=True
        )

        # Stop tokens for controlled generation
        stop_words = [".", "\n"]
        stop_token_ids = [self.tokenizer.encode(word, add_special_tokens=False)[-1] for word in stop_words]
        self.stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
        
        utils.log(f"Llama model initialized on: {self.device} with full float16 precision")
        
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
            "INTRO_SEQUENCE": "A"
        }

    def _create_prompt_templates(self):
        """Create optimized prompt templates for Llama"""
        return {
            "TITLE_SCREEN": "You're at the Pokémon Blue title screen. Respond with: 'Press START button to begin the game.'",
            "INTRO_SEQUENCE": "You're in Professor Oak's introduction. Respond with: 'Press A button to continue dialogue.'",
            "DIALOGUE": "You're in a dialogue screen. Respond with: 'Press A button to continue dialogue.'",
            "BATTLE": "You're in a Pokémon battle. Respond with exactly one of: 'Move UP', 'Move DOWN', or 'Press A'.",
            "MENU": "You're navigating a menu. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Press A', 'Press B'.",
            "OVERWORLD": "You're exploring the overworld. Respond with exactly one of: 'Move UP', 'Move DOWN', 'Move LEFT', 'Move RIGHT', 'Press A'."
        }

    def detect_game_state(self, game_state_text):
        """
        Analyzes the game state description to determine the player's situation.
        """
        game_state_upper = game_state_text.upper()

        if "BLUE VERSION" in game_state_upper or "TITLE SCREEN" in game_state_upper:
            return "TITLE_SCREEN"
        if "BATTLE" in game_state_upper or "FIGHT" in game_state_upper:
            return "BATTLE"
        if "TEXT" in game_state_upper or "DIALOGUE" in game_state_upper:
            return "DIALOGUE"
        if "OAK" in game_state_upper or "PROFESSOR" in game_state_upper:
            return "INTRO_SEQUENCE"
        if "MENU" in game_state_upper or "OPTION" in game_state_upper:
            return "MENU"
        
        return "OVERWORLD"

    def get_llama_instruction(self, game_state):
        """
        Uses Llama-3.2-3B to generate a concise instruction based on the current game state.
        """
        new_state = self.detect_game_state(game_state)
        if new_state != self.current_state:
            self.state_transitions += 1
            utils.log(f"State transition: {self.current_state} -> {new_state}")
            self.current_state = new_state

        # Direct action mapping for common states
        if new_state in self.common_state_actions:
            return f"Press {self.common_state_actions[new_state]} button."

        # Use state-based prompt
        prompt = self.prompt_templates.get(new_state, self.prompt_templates["OVERWORLD"])
        utils.log(f"Generating instruction for state: {new_state}")

        # Convert prompt into input tensor with float16 dtype
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device).to(torch.float16)

        # Generate response
        with torch.inference_mode():
            output = self.llama_model.generate(
                input_ids, 
                max_new_tokens=25,
                temperature=0.5,
                top_p=0.85,
                repetition_penalty=1.1,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode output
        instruction = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        utils.log(f"Llama instruction: {instruction}")

        return instruction if instruction else "Press A button to continue."

    def get_next_action(self):
        """
        Uses Vision to obtain the game state and determine the next button press.
        """
        game_state = self.vision.get_game_state_text()
        utils.log(f"Game state from vision: {game_state}")

        if "TITLE SCREEN" in game_state.upper():
            return "START"

        # Generate task instruction
        instruction = self.get_llama_instruction(game_state)
        utils.log(f"Llama instruction: {instruction}")

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
        
        return self.get_next_action()  # Recursive call to try again for AI PLAYTHROUGH
