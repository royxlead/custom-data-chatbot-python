# Custom Data Chatbot (Python)

A lightweight **chatbot** that uses **Python**, **NLTK**, and **JSON-based intents** to understand user input and respond accordingly. Great for kickstarting conversational AI projects with your own training data.

## Features

* **Customizable intents** via `intents.json`.
* **Simple natural language processing** with NLTK and Python.
* **Train and chat** workflow: easy training followed by interactive chatting.
* **Extendable**—add new intents, improve response logic, and enhance conversational behavior.

## How to Get Started

### 1. Clone the Repository

```bash
git clone https://github.com/royxlead/custom-data-chatbot-python.git
cd custom-data-chatbot-python
```

### 2. Install Dependencies

Requires Python 3.8+ (recommended) and the project's Python packages. Install them with:

```bash

# Custom Data Chatbot (Python)

A lightweight chatbot that uses Python, NLTK and a small Keras model to match user messages to intents defined in `intents.json`.

This repository provides a minimal train -> serve workflow so you can customize intents and run a local interactive bot.

## Highlights

- Easy-to-edit intent definitions in `intents.json`.
- Training script `training.py` that produces `words.pkl`, `classes.pkl`, `chatbot_model.h5` and `chatbotmodel_weights.h5`.
- Interactive runner `chatbot.py` that loads artifacts and serves responses on the command line.

## Requirements

- Python 3.8+ recommended
- See `requirements.txt` for Python package dependencies (TensorFlow, NLTK, NumPy).

## Quickstart (Windows PowerShell)

1. Create and activate a virtual environment, install packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2. Download required NLTK data (scripts will also attempt to download if missing):

```powershell
python - <<'PY'
import nltk
nltk.download('punkt')
nltk.download('wordnet')
print('NLTK data ready')
PY
```

3. Train the model (creates artifacts used by `chatbot.py`):

```powershell
python .\training.py --epochs 200 --batch-size 5
```

4. Run the interactive chatbot:

```powershell
python .\chatbot.py
# or with debug logging
python .\chatbot.py --verbose
```

## Files produced by training

- `words.pkl` — token vocabulary
- `classes.pkl` — intent labels
- `chatbot_model.h5` — full saved Keras model (preferred load)
- `chatbotmodel_weights.h5` — weights file
- `training_history.pkl` — training loss/accuracy history

These files are added to `.gitignore` by default to avoid committing large artifacts.

## Customizing intents

Edit `intents.json` to add or modify intents. Each intent should have:

- `tag` — unique identifier
- `patterns` — example user messages
- `responses` — possible bot replies (one is selected randomly)

After editing `intents.json`, re-run `training.py` to update model artifacts.

## Troubleshooting

- "NLTK data not found" — run the NLTK download step above or allow the scripts to download when they run.
- "Model or artifacts missing" — run `training.py` and ensure `words.pkl` and `classes.pkl` exist in the repo root.
- If TensorFlow installation fails, try a CPU-only install or a compatible TensorFlow wheel for your Python version.

## Next steps / enhancements

- Add unit/smoke tests that validate `intents.json` (every intent has at least one pattern and response).
- Add a small HTTP wrapper (Flask/FastAPI) to expose the bot over REST.
- Improve NLP with sentence embeddings (semantic matching) or replace the classifier with a transformer-based model.

## Contributing

Small fixes and intent improvements are welcome. For larger changes (model architecture, new backends), open an issue first.

---

If you'd like, I can add a small validator script that warns about placeholder links in `intents.json` or create a smoke test to validate the training & prediction flow. Reply with which you'd prefer and I'll add it.

