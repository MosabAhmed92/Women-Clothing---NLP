# Women's Clothing Reviews — End-to-End NLP Project

This repo follows a learning-first, no-copy-paste mentoring workflow. You will implement each step yourself, guided by hints and pseudo‑code.

## Problem (draft)
Predict (1) review **sentiment** (pos/neutral/neg), (2) **star rating** (1–5), and (3) **recommendation** (yes/no) from women's clothing reviews (text + structured features).

### Metrics
- Sentiment: F1 (macro), confusion matrix  
- Rating: weighted F1, confusion matrix  
- Recommendation: ROC‑AUC, PR curve, threshold analysis

## Repo Structure
```
project/
  data/raw/                # original data (read-only)
  data/processed/          # cleaned/engineered datasets
  notebooks/               # exploration & reports
  src/
    data/                  # loading, cleaning
    features/              # vectorization, feature union
    models/                # training, saving
    eval/                  # metrics, plots
    app/                   # simple API/Streamlit
  reports/                 # generated reports
  figs/                    # charts
  configs/                 # experiment configs (YAML)
```

## Getting Started (conda)
```bash
# create env
conda env create -f environment.yml
conda activate wcr-nlp

# register kernel
python -m ipykernel install --user --name wcr-nlp
```

> **Apple Silicon note:** PyTorch and TensorFlow both install via `pip` in this env. For PyTorch MPS support (Mac GPU), verify:  
> ```python
> import torch; torch.backends.mps.is_available()
> ```

## Phase 0 — Setup & Framing (you do this now)
1. Confirm dataset columns & choose initial target(s) for Phase 1.  
2. Write your 5–7 line problem statement in this README (replace the draft).  
3. Plan the **data audit**: missing values, duplicates, class balance, length distribution, and leakage (e.g., star mentions).  
4. Decide a **split policy**: stratified train/valid/test with fixed seed.

## Notes
- Do **not** commit `data/processed/` or large artifacts.  
- Keep experiments reproducible by saving configs to `configs/`.