# News Topic Classifier using BERT

A fine-tuned BERT (`bert-base-uncased`) model that classifies news headlines into one of four categories: **World**, **Sports**, **Business**, or **Sci/Tech**. Built and trained on Google Colab using the AG News dataset.

---

## 📌 Problem Statement

News articles need to be automatically categorized for tasks like content recommendation, content organization, and information retrieval. This project fine-tunes a pretrained BERT model to classify short news text (headline + description) into one of four predefined topic categories.

---

## 📂 Dataset

- **Name:** AG News
- **Source:** Hugging Face Datasets — [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news)
- **Classes (4):**
  - `0` — World
  - `1` — Sports
  - `2` — Business
  - `3` — Sci/Tech
- **Size used for training:** 8,000 training samples / 2,000 test samples (subset of full dataset for faster training on free-tier Colab GPU)
- **Fields:** `text` (headline + short description), `label` (0–3)

---

## 🛠️ Tech Stack / Libraries

| Library | Purpose |
|---|---|
| `transformers` | Pretrained BERT model + tokenizer + Trainer API |
| `datasets` | Loading and preprocessing AG News |
| `torch` | Backend deep learning framework |
| `scikit-learn` | Accuracy & F1-score evaluation metrics |
| `accelerate` | Required backend for Hugging Face `Trainer` |
| `gradio` | Quick interactive web demo / deployment |

---

## ⚙️ Environment Setup

1. Open the notebook in **Google Colab**
2. Go to **Runtime → Change runtime type → Hardware accelerator → GPU (T4)**
3. Install dependencies:

```python
!pip install -q -U transformers datasets evaluate scikit-learn gradio accelerate
```

> **Note:** If you hit a `VideoReader` import error from `torchvision`, upgrade `datasets` (`pip install -U datasets`) and restart the runtime — this is a known compatibility quirk in Colab's pre-installed packages, unrelated to this project's code.

*(Optional)* If you want higher Hugging Face Hub rate limits, add an `HF_TOKEN` secret in Colab (🔑 icon in sidebar → Secrets). Not required for public models/datasets like this one.

---

## 🧠 Methodology

1. **Load Data** — AG News dataset via Hugging Face `datasets`
2. **Tokenization** — `BertTokenizerFast` (`bert-base-uncased`), max length 64, padding + truncation
3. **Model** — `BertForSequenceClassification` with a 4-class classification head, fine-tuned end-to-end
4. **Training** — Hugging Face `Trainer` API, 2 epochs, batch size 16 (train) / 32 (eval)
5. **Evaluation** — Accuracy and weighted F1-score on the test set
6. **Deployment** — Interactive Gradio web demo for live predictions

---

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | *fill in from your `trainer.evaluate()` output* |
| F1-score (weighted) | *fill in from your `trainer.evaluate()` output* |

> Replace the above with your actual numbers from Step 09 (`trainer.evaluate()`) before submitting.

---

## 🚀 How to Run

1. Open the notebook in Google Colab
2. Run all cells in order (**Runtime → Run all**)
3. Once training finishes, the final cell launches a **Gradio demo** with a public shareable link
4. Enter any news headline into the textbox to get a live category prediction

Example:
```python
predict_news_category("Apple unveils new AI chip for its next-gen iPhones")
# Output: Sci/Tech
```

---

## 📁 Project Structure

```

├── notebook.ipynb              # Full Colab notebook (training + deployment)
└── README.md                   # This file
```

---

## 🔮 Future Improvements

- Train on the full AG News dataset (120k samples) instead of a subset for better accuracy
- Experiment with `distilbert-base-uncased` for a lighter, faster model
- Add a confusion matrix and per-class precision/recall breakdown
- Deploy permanently via Hugging Face Spaces instead of a temporary Gradio link

---

## 👤 Author

*Abdul Rafaih Mujeeb Abbasi*
Internship Task — AI/ML Engineering (Advanced Track)
