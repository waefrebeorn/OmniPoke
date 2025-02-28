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
from gamepad import Gamepad

class Emulator:
    def __init__(self):
        self.polling_interval = config.POLLING_INTERVAL
        self.window_region = None
        self.last_keypress_time = 0
        self.process = None
        self.gamepad = Gamepad()
        self.window_title = "mGBA"

        self.attach_or_launch_emulator()
        success = self.update_window_region()
        if not success:
            utils.log("WARNING: Failed to detect mGBA window. Capturing may not work correctly.")

    def is_emulator_running(self):
        if self.process is None:
            return False
        return self.process.poll() is None if hasattr(self.process, 'poll') else self.process.is_running()

    def find_existing_emulator(self):
        for proc in psutil.process_iter(attrs=["pid", "name"]):
            if "mgba" in proc.info["name"].lower():
                utils.log(f"Attached to existing mGBA process (PID {proc.info['pid']})")
                return psutil.Process(proc.info["pid"])
        return None

    def update_window_region(self):
        """
        Detect mGBA window, crop out top menu bar, etc.
        """
        try:
            windows = gw.getWindowsWithTitle(self.window_title)
            if windows:
                window = windows[0]
                if window.isMinimized:
                    win32gui.ShowWindow(window._hWnd, win32con.SW_RESTORE)
                    time.sleep(0.5)
                
                x, y, width, height = window.left, window.top, window.width, window.height

                # Approx offsets
                top_menu_height = 120
                bottom_border = 110
                side_border = 75

                content_x = x + side_border
                content_y = y + top_menu_height
                content_width = width - (side_border * 2)
                content_height = height - top_menu_height - bottom_border

                self.window_region = (content_x, content_y, content_width, content_height)
                utils.log(f"Updated mGBA window region to: {self.window_region}")

                self.debug_capture()
                return True
            else:
                utils.log("No mGBA window found to update region.")
                return False
        except Exception as e:
            utils.log(f"Error updating window region: {e}")
            return False

    def debug_capture(self):
        if self.window_region is None:
            utils.log("No window region to capture.")
            return
        try:
            screenshot = pyautogui.screenshot(region=self.window_region)
            debug_path = os.path.join(config.FRAME_SAVE_DIR, "window_debug.png")
            screenshot.save(debug_path)
            utils.log(f"Saved debug capture: {debug_path}")
        except Exception as e:
            utils.log(f"Failed to save debug capture: {e}")

    def bring_emulator_to_front(self):
        try:
            windows = gw.getWindowsWithTitle(self.window_title)
            if windows:
                window = windows[0]
                win32gui.ShowWindow(window._hWnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(window._hWnd)
                utils.log("Brought mGBA window to the front.")
                self.update_window_region()
            else:
                utils.log("No mGBA window found to bring to front.")
        except Exception as e:
            utils.log(f"Failed to bring mGBA to front: {e}")

    def attach_or_launch_emulator(self):
        existing_process = self.find_existing_emulator()
        if existing_process:
            self.process = existing_process
        else:
            self.launch_emulator()

    def launch_emulator(self):
        if not self.is_emulator_running():
            utils.log("Launching mGBA emulator...")
            self.process = subprocess.Popen([config.EMULATOR_PATH, config.ROM_PATH])
            time.sleep(3)
            self.bring_emulator_to_front()

    def restart_emulator_if_closed(self):
        if self.process is None or (hasattr(self.process, 'is_running') and not self.process.is_running()):
            utils.log("Emulator closed unexpectedly. Restarting...")
            self.launch_emulator()

    def capture_screen(self):
        self.restart_emulator_if_closed()
        self.bring_emulator_to_front()

        if not self.update_window_region():
            utils.log("ERROR: Failed to update window region for capture.")
            if self.window_region:
                black_frame = np.zeros((self.window_region[3], self.window_region[2], 3), dtype=np.uint8)
                return black_frame
            else:
                return np.zeros((320, 288, 3), dtype=np.uint8)  # default fallback

        screenshot = pyautogui.screenshot(region=self.window_region)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    def press_key(self, key):
        current_time = time.time()
        if current_time - self.last_keypress_time < 0.3:
            return
        self.restart_emulator_if_closed()
        self.bring_emulator_to_front()
        pyautogui.press(key)
        self.last_keypress_time = current_time
        utils.log(f"Pressed key: {key}")

    def press_button(self, action):
        """
        Use the virtual gamepad to press a button.
        """
        self.restart_emulator_if_closed()
        self.bring_emulator_to_front()
        self.gamepad.press_button(action)
        utils.log(f"Pressed gamepad button: {action}")

    def trigger_screenshot(self):
        utils.log("Triggering mGBA screenshot via gamepad...")
        self.gamepad.press_button("SCREENSHOT")

if __name__ == "__main__":
    emulator = Emulator()
    utils.log("Emulator test mode: capturing frames every 5 seconds.")
    utils.log(f"Initial window region: {emulator.window_region}")
    count = 0
    while count < 5:
        frame = emulator.capture_screen()
        utils.save_frame(frame, f"test_frame_{count}.png")
        utils.log(f"Captured test frame {count}")
        count += 1
        time.sleep(5)
