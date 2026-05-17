# Marathi Word Sense Disambiguation using MuRIL

A supervised Word Sense Disambiguation system for Marathi built using fine-tuned MuRIL as a binary cross-encoder.

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 88.94% |
| Macro-F1 | 0.87 |
| Test Records | 7,542 |
| Words Covered | 16 |
| Senses Covered | 50 |

## Dataset

- 50,242 labeled sentence-sense pairs across 16 ambiguous Marathi words and 50 senses
- Scraped from 11 Marathi news portals using BeautifulSoup and Stanza
- Labeled using MuRIL token-level cosine similarity against seed anchor vectors
- Split: 70% train / 15% validation / 15% test (sentence-level stratified)

## Models Compared

| Model | Role |
|-------|------|
| MuRIL | Main model |
| IndicBERT | Baseline |
| FastText | Baseline |
## Project Structure

```
marathi_wsd/
├── main.py                      → FastAPI backend
├── index.htm                    → Frontend UI
├── requirements.txt             → Dependencies
├── .gitignore
└── marathi_wsd_deployment/
    └── marathi_wsd_deployment/
        ├── metadata.json        → Sense glosses and word-sense mappings
        ├── tokenizer/           → MuRIL tokenizer files
        ├── encoder_config/      → Encoder configuration
        ├── confusion_matrix.png → Evaluation results
        ├── training_curves.png  → Training graphs
        └── model_weights.pt     → Not included (file too large)
```
## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/rohitdhole241/marathi-wsd-muril.git
cd marathi-wsd-muril
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add model weights
Place `model_weights.pt` inside `marathi_wsd_deployment/marathi_wsd_deployment/`
Contact rohitdhole241@gmail.com to request the model file.

### 5. Start the API
```bash
uvicorn main:app --reload
```

### 6. Open the UI
Visit: http://localhost:8000/ui

## Tech Stack

| Area | Tools |
|------|-------|
| Model | MuRIL, IndicBERT, FastText |
| Framework | PyTorch, Hugging Face Transformers |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| NLP Tools | BeautifulSoup, Stanza, Gemini API |
| Language | Marathi (Devanagari Script) |
