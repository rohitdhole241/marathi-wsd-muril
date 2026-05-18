# Marathi Word Sense Disambiguation using MuRIL

Marathi is one of India's major languages with over 83 million speakers, yet it remains
severely under-resourced in NLP. One of the hardest unsolved problems in NLP is
**Word Sense Disambiguation (WSD)** — automatically figuring out which meaning of a
word is being used in a given sentence.

For example, the Marathi word **"नाव"** can mean three completely different things:
- **Name** → "त्याचे नाव रोहित आहे" (His name is Rohit)
- **Boat** → "नदीत नाव चालवत होते" (A boat was sailing in the river)
- **Reputation** → "त्याने चांगले नाव कमावले" (He earned a good reputation)

A human reader instantly understands the correct meaning from context. Teaching a
machine to do the same — especially for a morphologically complex language like Marathi
— is what this project solves.

This project builds a **complete end-to-end Marathi WSD system from scratch**:
- A hand-crafted dataset of 50,242 labeled records
- Three trained and compared models (MuRIL, IndicBERT, FastText)
- A working web interface for live predictions

---

## Final Results

| Metric | MuRIL (Ours) | IndicBERT | FastText |
|--------|-------------|-----------|----------|
| Test Accuracy | **88.94%** | baseline | baseline |
| Macro-F1 | **0.87** | baseline | baseline |
| Test Records | 7,542 | 7,542 | 7,542 |

| Dataset Stat | Value |
|-------------|-------|
| Total Records | 50,242 |
| Ambiguous Words | 16 |
| Total Senses | 50 |
| News Portals Scraped | 11 |
| Seed Sentences | 15 per sense |

---

## Why This Problem Is Hard

Marathi is a **morphologically rich agglutinative language** — a single root word
changes its form dramatically based on grammar:

```
Root: डोळा (eye)
Forms: डोळे, डोळ्याला, डोळ्यांना, डोळ्यांचे, डोळ्यांत, डोळ्यांमध्ये...
```

A naive approach that looks for exact word matches would miss most of these forms.
This project uses Stanford Stanza's lemmatizer to reduce every form back to its root
before matching — a critical design decision that makes the dataset complete and accurate.

---

## Complete Pipeline

```
Step 1: Web Scraping          → collect raw Marathi sentences from 11 news portals
Step 2: Seed Generation       → create 15 verified example sentences per sense via Gemini API
Step 3: Dataset Creation      → map scraped sentences to senses using MuRIL cosine similarity
Step 4: Dataset Splitting     → sentence-level stratified 70/15/15 split
Step 5: Model Training        → fine-tune MuRIL, IndicBERT, FastText on the dataset
Step 6: Deployment            → FastAPI backend + HTML frontend for live inference
```

---

## Step 1 — Web Scraping (`Dataset/Scrapping_data_.ipynb`)

Since **no publicly available Marathi WSD dataset exists**, the first step was to collect
raw naturally occurring Marathi text from the web.

**Sources scraped (11 portals):**

| Portal | Type |
|--------|------|
| Loksatta | News |
| Esakal | News |
| Lokmat | News |
| ABP Live Marathi | News |
| BBC Marathi | News |
| TV9 Marathi | News |
| News18 Marathi | News |
| Maharashtra Times | News |
| Mitraho | Blog |
| Marathi Spandan | Blog |
| Maayboli | Cultural |

**Process:**
1. For each portal homepage, extract article links containing keywords: `news`, `article`, `story`, or a 4-digit year
2. Scrape up to 100 article links per portal
3. Extract all paragraph text from each article
4. Discard articles under 300 characters (stubs, captions, navigation text)

**Cleaning pipeline:**
- Remove all URLs using regex
- Strip all non-Devanagari characters (English letters, symbols) — keep digits intentionally (needed for words like अंक)
- Collapse newlines into spaces
- Use **Indic NLP Library** sentence tokenizer for Marathi-aware sentence splitting
- Keep only sentences between **5 and 30 words** — short enough to be clear, long enough to have context
- Remove duplicate sentences

**Output:** `raw_marathi_sentences.txt`

---

## Step 2 — Seed Sentence Generation (`Seed_Sentence_gemeni.json`)

The seed file is the **heart of the labeling pipeline**. For every word-sense pair, we
need a set of high-quality example sentences that clearly and unambiguously show that
specific meaning of the word. These examples are used to create sense anchor vectors
for the automatic labeling in Step 3.

**How seeds were created:**
1. Drew sense definitions (glosses) from **Marathi WordNet**
2. For each sense, provided the gloss to **Gemini API** with a prompt asking it to
   generate 15 natural Marathi sentences showing that exact meaning
3. **Manually reviewed every generated sentence** — removed any that were ambiguous,
   grammatically awkward, or didn't clearly reflect the intended sense
4. Only sentences that passed review were kept

**Seed file structure:**
```json
{
  "डोळा": [
    {
      "sense_id": 101,
      "pos": "noun",
      "gloss": "आपल्या शरीराचा तो अवयव ज्याच्या मदतीने आपण बघतो (Eye / Physical Organ)",
      "sentences": [
        "तिचे डोळे खूप सुंदर आणि बोलके आहेत.",
        "धूळ गेल्यामुळे माझा उजवा डोळा लाल झाला आहे.",
        "सतत संगणकावर काम केल्याने त्याचे डोळे दुखू लागले.",
        ... 12 more sentences
      ]
    },
    {
      "sense_id": 102,
      "pos": "noun",
      "gloss": "बटाटा, ऊस किंवा इतर झाडांवर येणारा छोटा कोंब (Sprout / Node on a plant)",
      "sentences": [ ... 15 sentences ]
    },
    ... 3 more senses
  ],
  "मार्ग": [ ... ],
  ... 18 more words
}
```

**Complete sense inventory in seed file:**

| Word | POS | Senses | Sense Meanings (English) |
|------|-----|--------|--------------------------|
| डोळा | Noun | 5 | Eye, Plant sprout, Nap/Sleep, Surveillance, Millstone hole |
| मार्ग | Noun | 3 | Road/Path, Solution/Method, Philosophy/Ideology |
| वाट | Noun | 3 | Path/Route, Waiting/Expectation, Ruin/Destruction |
| वाटणे | Verb | 3 | To feel/think, To distribute/share, To grind/crush |
| देणे | Verb | 3 | To give/provide, To owe money, To permit/allow |
| भाव | Noun | 3 | Price/Rate, Emotion/Feeling, Importance/Respect |
| योग | Noun | 3 | Yoga/Exercise, Coincidence/Chance, Astrological destiny |
| अर्थ | Noun | 3 | Meaning/Significance, Finance/Wealth, Purpose/Usefulness |
| नाव | Noun | 3 | Name/Identity, Boat/Ship, Reputation/Fame |
| वर | Noun/Postp. | 3 | Above/On top of, Bridegroom/Husband, Divine blessing |
| बरोबर | Adj/Adv | 3 | Correct/Accurate, Together/With, Equal/Even |
| हलका | Adj | 3 | Light in weight, Mild/Gentle, Inferior/Low quality |
| मंद | Adj | 3 | Slow/Sluggish, Dim/Faint, Dull/Slow-witted |
| फोडणे | Verb | 3 | To break/crack, To reveal a secret, To create a rift |
| उडणे | Verb | 3 | To fly, To disappear/fade, To mock/ridicule |
| वार | Noun | 3 | Day of the week, Physical blow/strike, Unit of length |
| गंध | Noun | 3 | Smell/Fragrance, Sandalwood paste (ritual), A hint/trace |
| पाठ | Noun | 3 | Back of body, Lesson/Chapter, Memorization |
| अंक | Noun | 3 | Number/Digit/Score, Magazine issue/Edition, Drama act/Scene |
| गुण | Noun | 3 | Qualities/Virtues, Marks/Score, Medicinal effect/Cure |

> **Note:** 20 words were in the seed file. After dataset creation, 4 words (हलका, फोडणे, उडणे, गंध) were dropped because they had insufficient scraped sentences. Final dataset covers **16 words and 50 senses**.

---

## Step 3 — Dataset Creation (`Dataset/WSD_dataset_Murl.ipynb`)

This notebook takes the scraped sentences and automatically labels each one with the
correct sense using MuRIL embeddings and cosine similarity.

**Stage 1: Morphology-Aware Sentence Mapping**

Finding which scraped sentences contain a target word is not trivial due to Marathi's morphology.

```
Root word: डोळा
Sentence might contain: डोळे, डोळ्याला, डोळ्यांना, डोळ्यांत, डोळ्यांचे...
```

Solution:
1. Pass every scraped sentence through **Stanford Stanza** lemmatizer
2. Stanza reduces each word token to its root form
3. If the root form matches any of our 20 target words → sentence is a candidate
4. **Fallback:** if Stanza fails (encoding issues, unusual tokens) → prefix matching is used

**Stage 2: Automatic Sense Assignment using MuRIL Embeddings**

For each candidate sentence, we need to determine which sense of the target word
is being used. This is done purely through semantic similarity:

```
For each word-sense:
  1. Take the 15 seed sentences for that sense
  2. Pass each through MuRIL → extract token-level embedding for the target word
     (NOT sentence-level mean pooling — we want the specific word's context)
  3. Average the 15 embeddings → create one "anchor vector" for that sense

For each candidate sentence:
  1. Extract MuRIL token-level embedding for the target word
  2. Compute cosine similarity against every sense anchor vector
  3. If highest similarity ≥ 0.65 → assign that sense label
  4. If highest similarity < 0.65 → discard the sentence (too ambiguous)
  5. If any sense has fewer than 100 sentences → relax threshold to 0.60
     and add highest-scoring remaining sentences (padding step)
```

Why token-level instead of sentence-level embeddings?

> Sentence-level mean pooling blends together all words in the sentence. We only
> care about the contextual meaning of the **target word** itself, not the overall topic.
> Token-level extraction isolates exactly that.

**Stage 3: Binary Pairwise Labeling + Golden Seed Injection**

The dataset uses a **contrastive binary design** — for each sentence, one record is
created per sense of that word:

```
Sentence: "आज पेट्रोलचा भाव वाढला आहे"  (Today petrol prices rose)
Target word: भाव

Record 1: sentence + "Price/Rate" gloss          → label = 1 ✅ (correct)
Record 2: sentence + "Emotion/Feeling" gloss     → label = 0 ❌ (wrong)
Record 3: sentence + "Importance/Respect" gloss  → label = 0 ❌ (wrong)
```

This teaches the model to **discriminate** between senses — not just recognize the
correct one, but actively reject the wrong ones.

The verified seed sentences from Step 2 were also injected directly into the dataset
as **golden examples** — high-confidence records that anchor every sense even if the
scraping captured few natural sentences for that meaning.

**Final dataset stats:**

| Word | Senses | Total Records |
|------|--------|---------------|
| वर | 3 | 6,129 (most common postposition) |
| देणे | 3 | 3,539 |
| वाट | 3 | 791 |
| नाव | 3 | 600 |
| मार्ग | 3 | 537 |
| योग | 3 | 536 |
| अर्थ | 3 | 541 |
| पाठ | 3 | 487 |
| डोळा | 5 | 494 |
| भाव | 3 | 457 |
| मंद | 3 | 445 |
| वाटणे | 3 | 323 |
| बरोबर | 3 | 327 |
| वार | 3 | 574 |
| गुण | 3 | 345 |
| अंक | 3 | 293 |
| **Total** | **50** | **50,242** |

Class imbalance ratio: ~1 positive to 2 negative records per sentence
(structural feature of the contrastive design, not a data collection problem)

---

## Step 4 — Dataset Splitting (`Dataset/Dataset_Split.ipynb`)

**Why not just split rows randomly?**

Each sentence contributes multiple rows (one per sense). A random row-level split
would put some rows from the same sentence in training and others in testing —
the model would have already seen that sentence during training. This is **data leakage**.

**How we split instead:**
1. Group all rows by (sentence, target_word) — keep them together as one unit
2. Find the correct sense (label=1) for each group — use it as stratification key
3. **Stratified split** on groups: 70% train, 30% temp
4. Split temp equally: 15% validation, 15% test
5. **Rare senses** (only 1 sentence in whole dataset, e.g., डोळा sense 105 and वाटणे sense 403) → forced directly into training before stratification

**Final split sizes:**

| Split | Positive (label=1) | Negative (label=0) | Total Rows |
|-------|-------------------|-------------------|------------|
| Train | 11,495 | 23,676 | **35,171** |
| Val | 2,459 | 5,070 | **7,529** |
| Test | 2,464 | 5,078 | **7,542** |
| **Total** | **16,418** | **33,824** | **50,242** |

---

## Step 5 — Model Training

### MuRIL — Main Model (`Models/Murl_WSD_train_test (1).ipynb`)

MuRIL (Multilingual Representations for Indian Languages) was developed by Google
and pretrained on 22 Indian languages. It was chosen over alternatives because it
produces significantly more coherent and contextually sensitive representations for
Marathi, especially for morphologically complex tokens.

**How the model works:**

```
Input:
  [CLS] Marathi sentence [SEP] Sense definition (gloss) [SEP]

MuRIL Encoder (768-dim hidden states)
  ↓
CLS token vector (768-dim) — represents combined context of sentence + gloss
  ↓
Projection Head:
  Linear(768 → 512) → GELU → Dropout(0.2) → Linear(512 → 256)
  ↓
Classifier:
  Linear(256 → 2)
  ↓
Output: probability of [label=0 (no match), label=1 (match)]
```

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Base model | `google/muril-base-cased` |
| Max sequence length | 128 tokens |
| Batch size | 8 |
| Epochs | 8 |
| Learning rate | 2e-5 |
| Optimizer | AdamW (weight decay 0.01) |
| Scheduler | Linear warmup (10% of total steps) |
| Loss function | Weighted CrossEntropyLoss |
| Gradient clipping | 1.0 |
| Device | CUDA (GPU) |

**Why weighted loss?**

The dataset has ~2× more label=0 records than label=1. Without correction, the model
would learn to predict "no match" for everything and still get 66% accuracy. The weight
for label=1 is set to `(avg_senses - 1)` to compensate.

**Training result:** Best checkpoint saved based on Validation Macro-F1

```
Final Test Accuracy : 88.94%
Final Test Macro-F1 : 0.87
```

---

### IndicBERT — Baseline 1 (`Models/Final_indicbert_train_test.ipynb`)

IndicBERT (`ai4bharat/indic-bert`) is an ALBERT-based model trained on 12 Indian
languages. It is smaller and faster than MuRIL.

- **Same architecture** as MuRIL (768 → 512 → 256 → 2)
- **Same training config** except batch size = 16 (smaller model allows larger batches)
- **Same dataset** — identical train/val/test splits
- Trained purely as a **fair comparison baseline** against MuRIL

---

### FastText — Baseline 2 (`Models/Final_fasttext-train-test.ipynb`)

FastText is a completely different approach — no transformers, no contextual embeddings.
It uses pretrained Marathi word vectors to classify the most likely sense.

- Uses `cc.mr.300.vec` — pretrained Common Crawl Marathi vectors (300-dim, ~600MB)
- **Multi-class classifier** (not binary like BERT models)
- Only trains on label=1 rows (correct sense examples)
- Input format: `__label__<sense_id> <target_word> <sentence>`
- Word n-grams = 2 (captures some morphological patterns)
- Learning rate = 0.1, Dim = 300
- Evaluation done by asking FastText to predict sense from valid candidates only

FastText serves as a lightweight non-transformer baseline showing what is achievable
without contextual embeddings.

---

## Step 6 — Deployment (`main.py` + `index.html`)

**Backend (FastAPI):**
- Loads trained MuRIL model, tokenizer, and metadata at startup
- Accepts any Marathi sentence via POST `/predict`
- Automatically detects which known ambiguous word is present
- Scores every candidate sense by passing (sentence, gloss) pairs through the model
- Returns ranked sense predictions with confidence scores

**Frontend (HTML/CSS/JS):**
- Clean dark-themed UI with Devanagari font support
- Type or paste any Marathi sentence
- Animated confidence bar chart for all senses
- Three example sentences preloaded as quick-test chips
- Connects to the local FastAPI server

---

## Project Structure

```
## Project Structure

```
marathi_wsd/
│
├── Dataset/
│   ├── Scrapping_data_.ipynb              → Web scraping from 11 Marathi news portals
│   ├── WSD_dataset_Murl.ipynb             → Dataset creation: Stanza mapping + MuRIL labeling
│   └── Dataset_Split.ipynb                → Sentence-level stratified 70/15/15 split
│
├── Models/
│   ├── Murl_WSD_train_test (1).ipynb      → MuRIL fine-tuning (main model, 87.89%)
│   ├── Final_indicbert_train_test.ipynb   → IndicBERT fine-tuning (baseline 1)
│   └── Final_fasttext-train-test.ipynb    → FastText training (baseline 2)
│
├── Raw_marathi_sentences/
│   ├── raw_marathi_sentences (1).txt      → Scraped raw Marathi text batch 1
│   ├── raw_marathi_sentences_2.txt        → Scraped raw Marathi text batch 2
│   ├── raw_marathi_sentences_3.txt        → Scraped raw Marathi text batch 3
│   ├── raw_marathi_sentences_4.txt        → Scraped raw Marathi text batch 4
│   ├── raw_marathi_sentences_5.txt        → Scraped raw Marathi text batch 5
│   ├── raw_marathi_sentences_6.txt        → Scraped raw Marathi text batch 6
│   ├── raw_marathi_sentences_7.txt        → Scraped raw Marathi text batch 7
│   ├── raw_marathi_sentences_8.txt        → Scraped raw Marathi text batch 8
│   ├── raw_marathi_sentences_9.txt        → Scraped raw Marathi text batch 9
│   ├── raw_marathi_sentences_10.txt       → Scraped raw Marathi text batch 10
│   ├── raw_marathi_sentences_11.txt       → Scraped raw Marathi text batch 11
│   ├── raw_marathi_sentences_12.txt       → Scraped raw Marathi text batch 12
│   ├── raw_marathi_sentences_13.txt       → Scraped raw Marathi text batch 13
│   └── raw_marathi_sentences.txt          → Combined/final raw sentences file
│
├── marathi_wsd_deployment/
│   └── marathi_wsd_deployment/
│       ├── encoder_config/                → MuRIL encoder architecture config
│       ├── tokenizer/                     → MuRIL tokenizer (vocab + config)
│       ├── confusion_matrix.png           → Test set confusion matrix
│       ├── metadata.json                  → All sense glosses + word-sense mappings
│       ├── model_weights.pt               → Not included (908 MB — too large for GitHub)
│       └── training_curves.png            → Train vs validation loss per epoch
│
├── Seed_Sentence_gemeni.json              → 15 verified seed sentences per sense (20 words, 61 senses)
├── marathi_wsd_dataset_filtered (1).json  → Final labeled dataset (50,242 records)
├── best_muril_wsd.pt                      → Not included (908 MB — too large for GitHub)
├── index.html                             → Web UI for live predictions
├── LICENSE                                → MIT License
├── main.py                                → FastAPI inference backend
├── requirements.txt                       → Python dependencies
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/rohitdhole241/marathi-wsd-muril.git
cd marathi-wsd-muril
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add model weights
Place `model_weights.pt` inside:
```
marathi_wsd_deployment/marathi_wsd_deployment/model_weights.pt
```
> The model file is 908 MB and cannot be hosted on GitHub.
> Contact **rohitdhole241@gmail.com** to request the weights file.

### 5. Start the backend
```bash
uvicorn main:app --reload
```

### 6. Open the UI
Visit **http://localhost:8000/ui** in your browser.

### 7. Test the API directly (optional)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sentence": "आज पेट्रोलचा भाव वाढला आहे"}'
```

---

## Tech Stack

| Area | Tool |
|------|------|
| Main Model | MuRIL (`google/muril-base-cased`) |
| Baseline 1 | IndicBERT (`ai4bharat/indic-bert`) |
| Baseline 2 | FastText (`cc.mr.300.vec`) |
| Deep Learning | PyTorch, Hugging Face Transformers |
| Web Scraping | BeautifulSoup, Requests |
| Lemmatization | Stanford Stanza (Marathi model) |
| Sentence Tokenization | Indic NLP Library |
| Seed Generation | Gemini API |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Language | Marathi (Devanagari Script) |

---

