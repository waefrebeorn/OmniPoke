import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config
import utils
from vision import Vision

class DecisionModel:
    def __init__(self):
        """
        Initialize the decision model.
        This uses LLaMA to decide which Game Boy button to press based on the 
        game state description provided by Moondream Vision.
        """
        self.vision = Vision()  # Now using Moondream for vision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.LLAMA_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map={"": self.device}
        )
        self.recent_moves = []

    def generate_action(self, game_state_description):
        """
        Generate the next action as a controller input.
        The prompt instructs LLaMA to consider the detailed Pokémon Blue game state 
        (menus, overworld, battle, etc.) and output a single button press.
        """
        llama_prompt = f"""
You are controlling a Game Boy emulator playing Pokémon Blue.
Your goal is to reach the Hall of Fame in as few moves as possible.
You may press only one button at a time: UP, DOWN, LEFT, RIGHT, A, B, START, or SELECT.

Game State Breakdown:
{game_state_description}

Recent moves: {self.recent_moves[-5:]}

Decide the next button to press. Reply with a single word:
UP, DOWN, LEFT, RIGHT, A, B, START, or SELECT.
"""
        inputs = self.tokenizer(llama_prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=1,  # Only one word output
                temperature=config.LLAMA_TEMPERATURE,
                top_k=config.LLAMA_TOP_K,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
        action = self.tokenizer.decode(output[0], skip_special_tokens=True).strip().upper()
        utils.log(f"LLaMA Raw Decision: {action}")

        # Validate action
        valid_buttons = {"UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"}
        if action not in valid_buttons:
            utils.log(f"Invalid action '{action}' received. Defaulting to 'A'.")
            action = "A"

        self.recent_moves.append(action)
        if len(self.recent_moves) > 10:
            self.recent_moves.pop(0)

        return action

    def get_next_action(self):
        """
        Obtain the current game state description from the Vision module,
        and use LLaMA to determine the next controller input.
        """
        game_state_description = self.vision.get_game_state()
        return self.generate_action(game_state_description)
