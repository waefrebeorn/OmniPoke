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
        self.failure_count = 0
        self.stuck_count = 0
        self.last_actions = []  # Track recent actions to detect stuck states

    def is_stuck(self):
        """Check if we're stuck in the same game state repeating the same actions"""
        if len(self.last_actions) < 10:
            return False
            
        if len(set(self.last_actions[-10:])) == 1:
            self.stuck_count += 1
            utils.log(f"Detected possible stuck state - same action {self.last_actions[-1]} repeated 10 times")
            return True
            
        return False

    def perform_recovery_action(self):
        """Attempt to recover from stuck states with a sequence of actions"""
        utils.log("Performing recovery sequence...")
        recovery_sequence = ["B", "START", "B", "A", "DOWN", "DOWN", "A", "B"]
        
        for action in recovery_sequence:
            utils.log(f"Recovery action: {action}")
            self.emulator.press_button(action)
            time.sleep(1)
            
        self.stuck_count = 0
        self.last_actions = []

    def run(self):
        utils.log("Starting Pokémon Blue AI training (All Moondream + Llama)...")
        
        utils.log("Pressing START button to begin game...")
        self.emulator.press_button("START")
        time.sleep(25)  # Increased wait time to 15 seconds
        
        while not self.game_completed:
            try:
                action = self.decision.get_next_action()
                self.last_actions.append(action)
                if len(self.last_actions) > 20:
                    self.last_actions.pop(0)
                
                self.emulator.press_button(action)
                self.action_count += 1
                
                if self.is_stuck() and self.stuck_count >= 3:
                    utils.log(f"Detected stuck state {self.stuck_count} times - attempting recovery")
                    self.perform_recovery_action()
                
                game_state_text = self.vision.get_game_state_text()
                if "Hall of Fame" in game_state_text:
                    utils.log(f"Game completed in {self.action_count} actions!")
                    self.game_completed = True
                
                if self.decision.current_state == "UNKNOWN":
                    time.sleep(config.POLLING_INTERVAL * 2)
                else:
                    time.sleep(config.POLLING_INTERVAL)
                    
            except Exception as e:
                self.failure_count += 1
                utils.log(f"Error during gameplay loop: {e}")
                if self.failure_count > 5:
                    utils.log("Too many consecutive failures. Restarting emulator...")
                    self.emulator.restart_emulator_if_closed()
                    self.failure_count = 0
                time.sleep(config.POLLING_INTERVAL * 2)

if __name__ == "__main__":
    PokemonTrainerAI().run()
