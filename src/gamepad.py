import pyvjoy
import time
import config
import utils

class Gamepad:
    def __init__(self):
        """Initialize virtual gamepad using vJoy."""
        self.j = pyvjoy.VJoyDevice(1)  # Use vJoy Device 1

        # Updated Gamepad Mapping
        self.button_map = {
            "A": 1,
            "B": 2,
            "START": 8,
            "SELECT": 7,
            "UP": 3,
            "DOWN": 4,
            "LEFT": 5,
            "RIGHT": 6,
            "SCREENSHOT": 9  # New button mapped for mGBA screenshot
        }

        self.reset()

    def reset(self):
        """Reset all buttons on the virtual gamepad."""
        self.j.reset()
        self.j.reset_buttons()
        utils.log("vJoy gamepad reset to default state.")

    def press_button(self, action, is_manual=False):
        """Press a mapped button using vJoy.
        
        If is_manual=True, wait 5 seconds before pressing.
        """
        if action in self.button_map:
            button_id = self.button_map[action]

            if is_manual:
                utils.log(f"Waiting 5 seconds before pressing: {action}")
                time.sleep(5)

            utils.log(f"Pressing gamepad button: {action} (vJoy Button {button_id})")
            self.j.set_button(button_id, 1)
            time.sleep(0.2)  # Hold duration
            self.j.set_button(button_id, 0)  # Release button

        else:
            utils.log(f"Invalid action '{action}' sent to gamepad.")

if __name__ == "__main__":
    gp = Gamepad()
    print("Manual Gamepad Test Mode: Enter button names to simulate input.")
    while True:
        action = input("Enter button (A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, SCREENSHOT) or 'exit': ").strip().upper()
        if action == "EXIT":
            break
        gp.press_button(action, is_manual=True)
