# OmniPoke: AI-Powered Pokemon Blue Trainer 🚀

OmniPoke is an AI-driven system that **autonomously plays Pokemon Blue**, optimizing for minimal moves to reach the Hall of Fame. It integrates **LLaMA 3.2** for decision-making and **Moondream2** for vision processing, ensuring a seamless hands-free gaming experience.

---

## 🚀 Features
- **Fully Autonomous** – AI plays the game from start to finish.
- **Optimized Decision-Making** – Uses **LLaMA 3.2 (3B)** to minimize unnecessary actions.
- **Advanced Vision Processing** – Extracts in-game text using **Moondream2**.
- **mGBA Emulator Integration** – Interacts with the game in real time.
- **Real-Time Logging** – Saves AI decisions and screen captures for debugging.
- **Fine-Tuning Ready** – Adapt LLaMA to improve AI strategy.

---

## 🛠 Setup Instructions

### 1️⃣ Install Python & Virtual Environment
Ensure **Python 3.10+** is installed. Then run:
```sh
setup.bat
```
This will:
- Create a virtual environment (`venv`).
- Install required dependencies (`torch`, `transformers`, `opencv-python`, etc.).
- Verify Git installation (auto-installs if missing).

---

### 2️⃣ Place Your Pokemon Blue ROM
Place your **`pokemon_blue.gb`** file inside the `roms/` folder:
```
📂 OmniPoke
│── roms/
│   └── pokemon_blue.gb  # Place ROM here
```
---

### 3️⃣ Install & Configure Emulator
The AI requires **mGBA** to play Pokemon Blue.
- **Download mGBA** → [https://mgba.io/downloads.html](https://mgba.io/downloads.html)
- **Set Emulator Path in `config.py`**:
  ```python
  EMULATOR_PATH = "C:/Path/To/mGBA/mGBA.exe"
  ```

---

### 4️⃣ Authenticate Hugging Face
The AI uses **Moondream2** (OCR) and **LLaMA 3.2** (Decision-making).
Authenticate with Hugging Face:
```sh
huggingface-cli login
```
Or manually add it in `config.py`:
```python
HF_TOKEN = "your_huggingface_token"
```
If Git is missing, install it via:
```sh
winget install --id Git.Git -e --source winget
```

---

## 🎮 Running OmniPoke
Start the AI with:
```sh
run.bat
```
This will:
1. **Launch mGBA** with Pokemon Blue.
2. **Extract game state** using Moondream2.
3. **Make AI-driven decisions** via LLaMA 3.2.
4. **Execute keypresses** in the emulator.
5. **Repeat until the game is completed!**

---

## ⚙️ Project Structure
```
📂 OmniPoke
│── setup.bat             # Installs dependencies
│── run.bat               # Runs the AI
│── requirements.txt       # Required libraries
│── .gitignore            # Excludes cache, logs
│── README.md             # Setup & usage guide
│── roms/
│   └── pokemon_blue.gb   # Pokemon Blue ROM (not included)
│── logs/                 # AI logs & debugging data
│── models/               # Cached Hugging Face models
│── src/
│   │── main.py          # AI logic & game loop
│   │── emulator.py      # Screen capture & key inputs
│   │── vision.py        # Uses Moondream2 for OCR
│   │── decision.py      # Uses LLaMA 3.2 for decisions
│   │── config.py        # Paths, settings, polling rates
│   │── utils.py         # Logging & helper functions
```

---

## 🔥 How It Works
1. **Vision Processing (Moondream2)** → Extracts text & game state.
2. **Decision Making (LLaMA 3.2)** → Chooses the optimal action.
3. **Execution (mGBA Emulator)** → Simulates keypresses.
4. **Loop Until Completion** → AI plays until the Hall of Fame.

---

## 🚀 Advanced: Fine-Tuning LLaMA 3.2
OmniPoke can be fine-tuned for improved Pokemon Blue strategies:
- Train **LLaMA 3.2** on past game runs.
- Adjust temperature & randomness in `config.py`:
  ```python
  LLAMA_TEMPERATURE = 0.7  # Lower = more deterministic
  LLAMA_MAX_TOKENS = 5     # Limits response length
  LLAMA_TOP_K = 5          # Restricts randomness
  ```

---

## 🛠 Troubleshooting

### **Emulator Won’t Launch?**
- Ensure `EMULATOR_PATH` is correct in `config.py`.
- Try running manually:
  ```sh
  C:/Path/To/mGBA/mGBA.exe roms/pokemon_blue.gb
  ```

### **Moondream2 Not Detecting Text?**
- Increase brightness/contrast in the emulator.
- Check `vision.py` logs for game state output.

### **Hugging Face Token Issues?**
- Run:
  ```sh
  huggingface-cli login
  ```
- If Git isn’t installed:
  ```sh
  winget install --id Git.Git -e --source winget
  ```

---

## 🎯 Next Steps
✔ **First AI test run** – Verify logging & action selection.  
✔ **Fine-tune LLaMA 3.2** – Improve AI’s strategy.  
✔ **Speedrun Optimization** – Train AI to reach TAS-level efficiency.  

---

🎮 **Ready to watch AI beat Pokemon Blue?** **Run `run.bat` and let it play!** 🚀

