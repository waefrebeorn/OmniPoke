import time
import utils
import config
import torch
from emulator import Emulator
from vision import Vision
from decision import Decision

class PokemonTrainerAI:
    def __init__(self):
        # Set PyTorch to use deterministic algorithms for better performance
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Initialize components
        self.emulator = Emulator()
        self.vision = Vision(self.emulator)
        self.decision = Decision(self.vision)

        self.action_count = 0
        self.game_completed = False
        self.failure_count = 0
        self.stuck_count = 0
        self.last_actions = []  # Track recent actions to detect stuck states
        self.last_action_time = time.time()
        
        # Track timing statistics for performance monitoring
        self.timing_stats = {
            "vision": [],
            "decision": [],
            "action": []
        }

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
            time.sleep(0.5)  # Reduced wait time for faster recovery
            
        self.stuck_count = 0
        self.last_actions = []

    def log_performance_stats(self):
        """Log performance statistics periodically"""
        if self.action_count % 50 == 0:
            stats = {}
            for key, values in self.timing_stats.items():
                if values:
                    stats[key] = {
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "total": len(values)
                    }
            
            utils.log(f"Performance stats after {self.action_count} actions:")
            for key, metrics in stats.items():
                utils.log(f"  {key}: avg={metrics['avg']:.3f}s, min={metrics['min']:.3f}s, max={metrics['max']:.3f}s")
            
            # Reset stats after logging
            self.timing_stats = {k: [] for k in self.timing_stats.keys()}

    def run(self):
        utils.log("Starting Pokémon Blue AI training with optimized models...")
        
        utils.log("Pressing START button to begin game...")
        self.emulator.press_button("START")
        time.sleep(5)  # Reduced wait time
        
        while not self.game_completed:
            try:
                start_time = time.time()
                
                # Execute next action
                decision_start = time.time()
                action = self.decision.get_next_action()
                decision_time = time.time() - decision_start
                self.timing_stats["decision"].append(decision_time)
                
                # Press the button and wait 1 second for the game to update
                self.emulator.press_button(action)
                time.sleep(1)  # Wait for the game to reflect the action
                
                # Capture updated screenshot after waiting
                frame = self.emulator.capture_screen()
                utils.save_frame(frame, "post_action.png")  # Overwrite the same file
                
                # Get game state text from the updated frame
                vision_start = time.time()
                game_state_text = self.vision.get_game_state_text()
                vision_time = time.time() - vision_start
                self.timing_stats["vision"].append(vision_time)
                
                # Add action to history
                self.last_actions.append(action)
                if len(self.last_actions) > 20:
                    self.last_actions.pop(0)
                
                self.action_count += 1
                
                # Log action time and potential performance issues
                elapsed = time.time() - self.last_action_time
                self.last_action_time = time.time()
                
                if self.action_count % 10 == 0:
                    utils.log(f"Action {self.action_count}: {action} - Vision: {vision_time:.3f}s, Decision: {decision_time:.3f}s, Total: {elapsed:.3f}s")
                    self.log_performance_stats()
                
                # Check for stuck state and recovery
                if self.is_stuck() and self.stuck_count >= 3:
                    utils.log(f"Detected stuck state {self.stuck_count} times - attempting recovery")
                    self.perform_recovery_action()
                
                # Check for game completion
                if "Hall of Fame" in game_state_text:
                    utils.log(f"Game completed in {self.action_count} actions!")
                    self.game_completed = True
                
                # Dynamic polling interval based on game state
                if self.decision.current_state == "UNKNOWN":
                    time.sleep(config.POLLING_INTERVAL * 1.5)
                elif self.decision.current_state in ["TITLE_SCREEN", "DIALOGUE", "INTRO_SEQUENCE"]:
                    time.sleep(config.POLLING_INTERVAL * 0.7)  # Faster for simple states
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
    # Try to free up memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    PokemonTrainerAI().run()