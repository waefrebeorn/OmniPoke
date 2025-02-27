import time
import utils
import config
from emulator import Emulator
from vision import Vision
from decision import Decision

class PokemonTrainerAI:
    def __init__(self):
        self.emulator = Emulator()         # For capturing screen & gamepad button presses
        self.vision = Vision(self.emulator)  # Moondream-based vision+decision
        self.decision = Decision(self.vision)

        self.action_count = 0
        self.game_completed = False

    def run(self):
        utils.log("Starting Pokémon Blue AI training (All Moondream)...")
        while not self.game_completed:
            # Get next action from Decision (which calls Vision -> Moondream)
            action = self.decision.get_next_action()
            # Use the gamepad to press the corresponding button
            self.emulator.press_button(action)
            self.action_count += 1

            # Optional: Check if we see "Hall of Fame" in the screen text
            game_state_text = self.vision.get_game_state_text()
            if "Hall of Fame" in game_state_text:
                utils.log(f"Game completed in {self.action_count} actions!")
                self.game_completed = True

            time.sleep(config.POLLING_INTERVAL)

if __name__ == "__main__":
    PokemonTrainerAI().run()
