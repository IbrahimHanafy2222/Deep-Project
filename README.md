# Grammatical Error Correction on C4_200M

Course project for **CIE 555 — Neural Networks and Deep Learning**.

A comparative study of four models for **Grammatical Error Correction** (GEC):
taking an ungrammatical English sentence and producing its corrected form.

```
Input  (corrupted): "She go to school yesterday and learning many thing."
Output (corrected):  "She went to school yesterday and learned many things."
```

## Team

| Name | ID | Role |
|------|----|------|
| Abdullah Ahmed | 202200206 | Author |
| Ibrahim Hanafy | 202200518 | Author |

Instructor: Dr. Ibrahim Swelam · TAs: Aya Abdelaziz, Mahmoud Farahat

## Dataset

**C4_200M** — synthetic GEC corpus (clean C4 web text + automatic corruption).
We use 1,000,000 `(corrupted, clean)` pairs:

| Split | Pairs |
|-------|-------|
| Train | 850,000 |
| Validation | 50,000 |
| Test | 100,000 |

Tokenizer: 16k Byte-Level BPE, max sequence length 64.

## Models

| # | Model | Notebook | Task | Params | Test GLEU |
|---|-------|----------|------|--------|-----------|
| 1 | TF-IDF Bag-of-Words | `04-bow-baseline` | Detection | — | ~62% acc |
| 2 | LSTM encoder–decoder | `05-lstm-seq2seq` | Correction | 19.55M | 0.213 |
| 3 | LSTM + Bahdanau attention | `06-lstm-bahdanau` | Correction | 29.06M | 0.599 / 0.618 beam |
| 4 | Transformer | `07-transformer` | Correction | 8.05M | 0.618 |

Best validation GLEU: **Transformer 0.613** (selected model).

## Results summary

- Adding **attention** is the decisive jump: +0.386 GLEU over the bottleneck LSTM.
- The **Transformer** matches the attention LSTM on test GLEU with 3.6× fewer
  parameters and far faster training.
- **Exact match ≈ 2%** for every model — a dataset ceiling: many C4_200M
  references are paraphrases or noisy, not minimal grammar corrections.

## Repository layout

```
notebooks/     01–07: data exploration → pipeline → tokenizer → BoW → LSTM → Bahdanau → Transformer
data/          tokenizer vocab/merges
evaluation/    phase4_bahdanau/ , phase5_transformer/ — predictions, summaries, attention plots
report/        report.tex (scientific report), figures/
presentation/  slides in Beamer (.tex), reveal.js (.html), and PowerPoint (.pptx)
requirements.txt
```

## Reproducing

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Run notebooks `01`→`07` in order. Notebooks were developed on Kaggle GPUs;
trained weights are not committed (rebuild from the notebooks).

## Building the report and slides

```bash
# Scientific report
cd report && pdflatex report.tex && pdflatex report.tex

# Presentation (Beamer PDF)
cd presentation && pdflatex slides.tex && pdflatex slides.tex
```

The PowerPoint deck (`presentation/slides.pptx`) and the reveal.js HTML deck
(`presentation/slides.html`) are also provided.

## Metrics

- **GLEU** — n-gram metric tuned for GEC; primary metric.
- **Exact Match** — strict, fraction of predictions equal to the reference.
- **Precision / Recall / F0.5** — token-level; F0.5 weights precision 2×.
