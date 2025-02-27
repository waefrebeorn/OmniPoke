import time
import utils
import config
from emulator import Emulator
from vision import Vision
from decision import DecisionModel

class PokemonTrainerAI:
    def __init__(self):
        # Initialize emulator first so we have window detection
        self.emulator = Emulator()
        
        # Initialize vision with reference to emulator for screen capture
        self.vision = Vision(self.emulator)
        
        # Decision model will also need vision with proper screen capture
        self.decision_model = DecisionModel()
        # Update the vision instance in the decision model
        self.decision_model.vision = self.vision
        
        self.action_count = 0
        self.game_completed = False

    def run(self):
        """Main AI loop that sends controller input based on AI decisions."""
        utils.log("Starting Pokémon Blue AI training...")
        while not self.game_completed:
            # Get game state description using vision module that uses emulator's window detection
            game_state_description = self.vision.get_game_state()
            action = self.decision_model.get_next_action()
            keypress = config.KEY_MAPPING.get(action)
            if keypress:
                self.emulator.press_key(keypress)
                self.action_count += 1
            if "Hall of Fame" in game_state_description:
                utils.log(f"Game completed in {self.action_count} actions!")
                self.game_completed = True
                break
            time.sleep(config.POLLING_INTERVAL)

if __name__ == "__main__":
    PokemonTrainerAI().run()