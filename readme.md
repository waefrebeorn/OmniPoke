# OmniPoke: AI-Powered Pokémon Blue Trainer 🚀

OmniPoke is an AI-driven system that **autonomously plays Pokémon Blue**—all powered by Moondream for vision processing and decision-making. The system is designed to optimize gameplay by reducing unnecessary moves and reaching the Hall of Fame as efficiently as possible.

> **Note:** This release is now entirely based on Moondream (both for vision and decision-making), replacing previous LLaMA 3.2 integrations.

---

## 🚀 Features
- **Fully Autonomous** – The AI plays the game from start to finish without human intervention.
- **Optimized Decision-Making** – Moondream analyzes the game screen and selects the best action.
- **Advanced Vision Processing** – In-game text and state are extracted using Moondream’s OCR capabilities.
- **mGBA Emulator Integration** – Interacts with Pokémon Blue in real time.
- **Virtual Gamepad Control** – Uses a virtual gamepad (via vJoy) to send inputs reliably.
- **Real-Time Logging & Debugging** – Saves AI decisions, screen captures, and logs for easy troubleshooting.
- **Fine-Tuning Ready** – Easily update settings in `config.py` for improved performance.

---

## 🛠 Setup Instructions

### 1️⃣ Install Python & Create Virtual Environment
Ensure you have **Python 3.10+** installed. Then run the setup script:
```sh
setup.bat
```
This script will:
- Create a virtual environment (`venv`).
- Install all required dependencies (such as `torch`, `transformers`, `opencv-python`, etc.).
- Verify or install Git if it is missing.

---

### 2️⃣ Place Your Pokémon Blue ROM
Copy your **`pokemon_blue.gb`** ROM file into the `roms/` folder:
```
📂 OmniPoke
│── roms/
│   └── pokemon_blue.gb  # Place your ROM file here
```

---

### 3️⃣ Install & Configure the mGBA Emulator
OmniPoke requires mGBA to run Pokémon Blue:
- **Download mGBA** from [mgba.io/downloads.html](https://mgba.io/downloads.html).
- In `config.py`, set the path to your mGBA executable:
  ```python
  EMULATOR_PATH = "C:/Path/To/mGBA/mGBA.exe"
  ```

---

### 4️⃣ Authenticate with Hugging Face
Moondream relies on models hosted on Hugging Face. Log in by running:
```sh
huggingface-cli login
```
Alternatively, add your token directly in `config.py`:
```python
HF_TOKEN = "your_huggingface_token"
```

---

## 🎮 Running OmniPoke

Start the AI by running:
```sh
run.bat
```
This batch file will:
1. Launch the mGBA emulator with Pokémon Blue.
2. Use `vision.py` (powered by Moondream) to capture and analyze the game screen.
3. Use `decision.py` to determine the best button to press.
4. Execute actions via `emulator.py` and `gamepad.py` using a virtual gamepad.
5. Continue until the game reaches the Hall of Fame.

---

## ⚙️ Project Structure & File Usage

```
📂 OmniPoke
│── setup.bat             # Script to install dependencies and setup the environment.
│── run.bat               # Script to launch the AI and start the game.
│── requirements.txt      # List of required Python libraries.
│── .gitignore            # Excludes logs, cache, and temporary files.
│── README.md             # This guide.
│── roms/
│   └── pokemon_blue.gb   # Your Pokémon Blue ROM.
│── logs/                 # Contains logs and captured frames for debugging.
│── models/               # Cached Hugging Face models.
│── src/
│   │── main.py          # Main AI loop that orchestrates the gameplay.
│   │── emulator.py      # Handles screen capturing and virtual gamepad input.
│   │── vision.py        # Uses Moondream to process screen images and decide actions.
│   │── decision.py      # Delegates decision-making (calls vision.py).
│   │── gamepad.py       # Virtual gamepad integration using vJoy.
│   │── config.py        # Contains configuration settings (paths, emulator options, etc.).
│   │── utils.py         # Logging and helper functions.
```

### File Overview & Examples

- **`main.py`**  
  Orchestrates the gameplay loop:
  - Captures the screen.
  - Queries Moondream (via `vision.py`) for the next move.
  - Sends the corresponding gamepad input using `emulator.py` and `gamepad.py`.
  
  **Example:**  
  ```python
  if __name__ == "__main__":
      PokemonTrainerAI().run()
  ```

- **`emulator.py`**  
  Manages the mGBA emulator instance:
  - Captures screen regions.
  - Sends keypresses via a virtual gamepad.
  - Ensures the emulator window is active and properly captured.

- **`vision.py`**  
  Handles vision and decision-making using Moondream:
  - Processes in-game screenshots.
  - Provides game state text (useful for debugging).
  - Crafts an improved prompt for Moondream to better handle title screens and menus.
  
  **Example Prompt in `vision.py`:**
  ```python
  moondream_prompt = f"""
  You are playing Pokémon Blue on a Game Boy.
  Analyze this screenshot carefully. If the screen shows the title screen or a menu, the best action is to press START to begin or continue the game.
  Otherwise, decide the single best button to press from the following options: {", ".join(self.valid_buttons)}.
  Return ONLY one of these words (UP, DOWN, LEFT, RIGHT, A, B, START, SELECT) with no extra text.
  """
  ```

- **`decision.py`**  
  Acts as a simple pass-through layer that:
  - Calls `vision.get_next_action()`.
  - Logs the chosen action before returning it.

- **`gamepad.py`**  
  Interfaces with vJoy to simulate gamepad inputs:
  - Maps actions (e.g., "A", "START") to specific vJoy buttons.
  - Provides both automated and manual testing modes for inputs.

- **`config.py` & `utils.py`**  
  - **`config.py`** centralizes all settings (paths, emulator details, logging options).
  - **`utils.py`** provides helper functions for logging and frame saving.

---

## 🔥 How It Works
1. **Vision Processing (Moondream):**  
   Captures the emulator screen and extracts in-game text and state.

2. **Decision Making (Moondream):**  
   Based on the image, Moondream selects the optimal button to press—especially recognizing title screens and menus where START is the best choice.

3. **Execution (mGBA & Virtual Gamepad):**  
   The selected action is executed using a virtual gamepad interface to interact with the mGBA emulator.

4. **Game Loop:**  
   The system continuously repeats this process until it detects that the game has reached the Hall of Fame.

---

## 🚀 Advanced: Fine-Tuning Moondream
While OmniPoke is now entirely powered by Moondream, you can still tweak settings in `config.py` if necessary. For example, adjust polling intervals or change logging levels. These settings help optimize decision-making and game speed.

---

## 🛠 Troubleshooting

### **Emulator Won’t Launch?**
- Double-check the `EMULATOR_PATH` in `config.py`.
- Try launching mGBA manually:
  ```sh
  C:/Path/To/mGBA/mGBA.exe roms/pokemon_blue.gb
  ```

### **Moondream Not Detecting Text?**
- Adjust screen brightness/contrast within the emulator.
- Check logs in `logs/` and review the output from `vision.py`.

### **Virtual Gamepad Issues?**
- Ensure vJoy is installed and properly configured.
- Review logs from `gamepad.py` to diagnose any mapping issues.

---

## 🎯 Next Steps
- **Initial Test Run:** Verify logs and action selections are working.
- **Optimize Decision-Making:** Monitor Moondream’s responses to fine-tune the prompt.
- **Expand Debugging:** Use logs and captured frames in the `logs/` directory to further refine performance.
- **Watch the AI in Action:** Run `run.bat` and see OmniPoke conquer Pokémon Blue!

---

Ready to watch the AI play Pokémon Blue? **Run `run.bat` and let OmniPoke do the rest!** 🚀


