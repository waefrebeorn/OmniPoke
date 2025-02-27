import pyautogui
import pygetwindow as gw
import time
import cv2
import numpy as np
import config
import utils
import subprocess
import psutil
import win32gui
import win32con
import os
from gamepad import Gamepad  # Virtual gamepad control

class Emulator:
    def __init__(self):
        self.polling_interval = config.POLLING_INTERVAL
        self.window_region = None  # No initial value - will be set by detection
        self.last_keypress_time = 0
        self.process = None
        self.gamepad = Gamepad()  # Virtual controller support
        self.window_title = "mGBA"  # Window title to search for

        # Ensure only one emulator instance runs
        self.attach_or_launch_emulator()
        
        # Set window region after launching
        success = self.update_window_region()
        if not success:
            utils.log("WARNING: Failed to detect mGBA window. Capturing may not work correctly.")

    def is_emulator_running(self):
        """Checks if the emulator process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None if hasattr(self.process, 'poll') else self.process.is_running()

    def find_existing_emulator(self):
        """Finds and attaches to an existing mGBA process if running."""
        for proc in psutil.process_iter(attrs=["pid", "name"]):
            if "mgba" in proc.info["name"].lower():
                utils.log(f"Attached to existing mGBA process (PID {proc.info['pid']})")
                return psutil.Process(proc.info["pid"])
        return None

    def update_window_region(self):
        """Detects and updates the mGBA window position and dimensions. 
        Returns True if successful, False otherwise."""
        try:
            windows = gw.getWindowsWithTitle(self.window_title)
            if windows:
                window = windows[0]
                # Check if window is minimized
                if window.isMinimized:
                    win32gui.ShowWindow(window._hWnd, win32con.SW_RESTORE)
                    time.sleep(0.5)  # Give time for the window to restore
                
                # Get window position and size
                x, y, width, height = window.left, window.top, window.width, window.height
                
                # Adjust for window borders and title bar (approximate)
                # These values might need adjustment based on Windows theme and scaling
                content_x = x + 8  # Adjust for left border
                content_y = y + 32  # Adjust for title bar
                content_width = width - 16  # Adjust for left and right borders
                content_height = height - 40  # Adjust for title bar and bottom border
                
                self.window_region = (content_x, content_y, content_width, content_height)
                utils.log(f"Updated mGBA window region to: {self.window_region}")
                
                # Save a screenshot for debugging
                self.debug_capture()
                return True
            else:
                utils.log("No mGBA window found to update region.")
                return False
        except Exception as e:
            utils.log(f"Error updating window region: {e}")
            return False

    def debug_capture(self):
        """Capture and save the current window region for debugging."""
        if self.window_region is None:
            utils.log("Cannot save debug capture: No window region detected.")
            return
            
        try:
            screenshot = pyautogui.screenshot(region=self.window_region)
            debug_path = os.path.join(config.FRAME_SAVE_DIR, "window_debug.png")
            screenshot.save(debug_path)
            utils.log(f"Saved debug capture of window region: {debug_path}")
        except Exception as e:
            utils.log(f"Failed to save debug capture: {e}")

    def bring_emulator_to_front(self):
        """Brings the mGBA window to the foreground to ensure key presses register."""
        try:
            windows = gw.getWindowsWithTitle(self.window_title)
            if windows:
                window = windows[0]
                win32gui.ShowWindow(window._hWnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(window._hWnd)
                utils.log("Brought mGBA window to the front.")
                # Update window region after bringing to front
                self.update_window_region()
            else:
                utils.log("No mGBA window found to bring to front.")
        except Exception as e:
            utils.log(f"Failed to bring mGBA to front: {e}")

    def attach_or_launch_emulator(self):
        """Attaches to an existing emulator process or launches a new one."""
        existing_process = self.find_existing_emulator()
        if existing_process:
            self.process = existing_process
        else:
            self.launch_emulator()

    def launch_emulator(self):
        """Launches the emulator with the Pokémon Blue ROM."""
        if not self.is_emulator_running():
            utils.log("Launching mGBA emulator...")
            self.process = subprocess.Popen([config.EMULATOR_PATH, config.ROM_PATH])
            time.sleep(3)  # Give emulator time to start
            self.bring_emulator_to_front()

    def restart_emulator_if_closed(self):
        """Restarts the emulator if it was closed."""
        if self.process is None or (isinstance(self.process, psutil.Process) and not self.process.is_running()):
            utils.log("Emulator closed unexpectedly. Restarting...")
            self.launch_emulator()

    def capture_screen(self):
        """Captures the emulator screen within the defined region using PyAutoGUI."""
        self.restart_emulator_if_closed()
        
        # Always update window region before capture
        if not self.update_window_region():
            utils.log("ERROR: Failed to update window region for screen capture")
            # Return a black frame if we can't detect the window
            if self.window_region:
                # Create a black frame matching the last known dimensions
                black_frame = np.zeros((self.window_region[3], self.window_region[2], 3), dtype=np.uint8)
                return black_frame
            else:
                # Create a default size black frame if we never had a valid window region
                utils.log("No valid window region has been detected yet")
                return np.zeros((320, 480, 3), dtype=np.uint8)
                
        # Capture the screen now that we have a valid window region
        screenshot = pyautogui.screenshot(region=self.window_region)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    def press_key(self, key):
        """Simulates pressing a key in the emulator, with a delay to prevent spam."""
        current_time = time.time()
        if current_time - self.last_keypress_time < 0.3:  # Prevents rapid key spam
            return
        
        self.restart_emulator_if_closed()
        self.bring_emulator_to_front()
        pyautogui.press(key)
        self.last_keypress_time = current_time
        utils.log(f"Pressed key: {key}")

    def trigger_screenshot(self):
        """Triggers an mGBA screenshot via the virtual gamepad."""
        utils.log("Triggering mGBA screenshot via gamepad...")
        self.gamepad.press_button("SCREENSHOT")

if __name__ == "__main__":
    emulator = Emulator()
    utils.log("Emulator test mode: Capturing frames every 5 seconds.")
    utils.log(f"Initial window region: {emulator.window_region}")
    count = 0
    while count < 5:  # Capture 5 frames for testing
        frame = emulator.capture_screen()
        utils.save_frame(frame, f"test_frame_{count}.png")
        utils.log(f"Captured test frame {count}")
        count += 1
        time.sleep(5)