# GestureGuru
### Interactive Indian Sign Language Learning System

A real-time desktop app that helps beginners **learn and practice Indian Sign Language (ISL)** with a webcam.

It recognizes **digits 1–9** and **letters A–Z**, and supports live detection, guided practice tests, keyboard typing from signs, and optional game gesture controls.

**Repo:** https://github.com/SirNutALot/GestureGuru-Interactive-Indian-Sign-Language-Learning-System

---

## Features

| Mode | What it does |
|------|----------------|
| **ISL Detection** | Live webcam view for checking / debugging recognition |
| **ISL Practice** | Shows a random letter or number; you sign it to score |
| **Keyboard Mapper** | Types recognized signs into the focused text field |
| **Game Gestures** | Extra hand controls for games (thumb WASD, palm cursor, fist click, etc.) |

---

## How it works

1. Webcam captures your hand  
2. **MediaPipe Hands** extracts 21 hand landmarks  
3. Landmarks are normalized into a feature vector  
4. A trained **TensorFlow / Keras** model (`models/model.h5`) classifies the sign  
5. The app shows feedback, scores practice attempts, or types the character  

---

## Requirements

- Windows (recommended; `setup.bat` / `run.bat` included)
- **Python 3.10 or 3.11** (TensorFlow 2.11 may fail on newer Python)
- Webcam
- Internet connection for first-time dependency install

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/SirNutALot/GestureGuru-Interactive-Indian-Sign-Language-Learning-System.git
cd GestureGuru-Interactive-Indian-Sign-Language-Learning-System
```

### 2. First-time setup

Double-click:

```text
setup.bat
```

This will:

- create a local `.venv` virtual environment  
- upgrade `pip`  
- install packages from `requirements.txt`  

### 3. Run the app

Double-click:

```text
run.bat
```

The launcher opens. Choose a mode and click **Launch**.

---

## Modes explained

### ISL Detection
Use this to verify the model is working. Hold a sign for **1–9** or **A–Z** and check the on-screen label.

### ISL Practice (learning / test)
A target character appears. Perform the matching ISL sign. Correct / wrong feedback is shown and your score updates.

### Keyboard Mapper
Stable recognized signs are typed as keyboard characters. Focus Notepad (or any text field) first, then sign.

### Game Gestures
Separate rule-based gesture controller (does not need `model.h5`).

---

## Controls

### Practice mode
- **ESC** — quit  
- **SPACE** — skip target  
- **R** — reset score  
- **T** — toggle target visibility  

### Keyboard mapper
- **ESC** — quit  
- **[** — toggle typing on/off  
- Focus a text field before signing  

### Detection
- **ESC** — quit  

### Game gestures
- **q** — quit  
- **c** — calibrate palm center  

---

## Project structure

```text
.
├── launcher.py          # Main GUI launcher
├── setup.bat            # First-time environment setup
├── run.bat              # Start the app
├── requirements.txt
├── README.md
├── models/
│   └── model.h5         # Trained ISL classifier
├── data/
│   └── keypoint.csv     # Landmark training data
├── src/
│   ├── common.py        # Shared preprocessing + model helpers
│   ├── isl_detection.py
│   ├── isl_practice.py
│   ├── isl_typer.py
│   └── game_control.py
├── training/            # Notebook + dataset generation tools
└── assets/              # Reference images
```

---

## Tech stack

- Python  
- OpenCV  
- MediaPipe  
- TensorFlow / Keras  
- NumPy, Pandas  
- Tkinter (launcher)  
- pynput / PyAutoGUI (typing and game input)  

---

## Training (optional)

Files in `training/`:

- `ISL_classifier.ipynb` — train the classifier  
- `dataset_keypoint_generation.py` — build landmark CSV from images  
- Dataset: `data/keypoint.csv`  
- Saved model path: `models/model.h5`  

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `setup.bat` fails on TensorFlow | Use Python **3.10** or **3.11**, then re-run setup |
| Camera not opening | Close other apps using the webcam; try unplugging/replugging |
| Model missing warning | Ensure `models/model.h5` exists in the project |
| Typing does nothing | Focus a text editor window, confirm typing is ON (`[`) |
| Wrong predictions | Improve lighting, keep hand clearly in frame, hold the sign steady |

---

## Notes

- Recognition coverage: **1–9** and **A–Z** (35 classes)  
- Digit **0** is not included  
- Use a clear background and good lighting for best results  
- Detection, Practice, and Keyboard Mapper require `models/model.h5`  

---

## License

Provided for educational / major-project use.
