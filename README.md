# Prediction Task — Medical Imaging

Research code for predicting clinical time-to-event outcomes from medical imaging using radiomics, pretrained vision models, and survival-analysis models.

The repository contains several experimental pipelines exploring:

- radiomics-based prediction;
- pretrained image representations;
- DINOv3 / ViT-based image embeddings;
- multimodal image + radiomics models;
- Cox survival models;
- multislice approaches;
- pretrained-backbone benchmarking;
- feature selection / pruning;
- Merlin-based (3D CT foundation model) experiments;
- nested cross-validation;
- model interpretation and confound analysis.

> **Research use only.**
> This repository is intended for research and experimentation. The models are not clinically validated and should not be used for medical decision-making.

---

## Table of contents

- [Overview](#overview)
- [TL;DR — where do I start reading?](#tldr--where-do-i-start-reading)
- [Repository structure](#repository-structure)
- [Data](#data)
- [Prediction targets](#prediction-targets)
- [Two model families](#two-model-families)
- [Config files](#config-files)
- [Outputs — how to tell them apart](#outputs--how-to-tell-them-apart)
- [Recommended way to navigate the code](#recommended-way-to-navigate-the-code)
- [Running things](#running-things)
- [Headline results](#headline-results)
- [Reproducibility notes](#reproducibility-notes)
- [Reference](#reference)

---

## Overview

The goal of this project is to use medical imaging to predict clinical outcomes, with particular emphasis on **time-to-event prediction**.

Rather than treating the problem as a simple binary classification task, the models use survival-analysis concepts so that both:

1. whether an event occurred, and
2. the time until the event or censoring

can be taken into account.

The repository implements several ways of representing the same medical images:

```text
                         Medical images
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
          Radiomics        DINOv3 / ViT     Multislice
              │                │                │
              ▼                ▼                ▼
       Hand-crafted       Learned image     Multiple image
         features           features          features
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                        Prediction model
                               │
                               ▼
                         Risk prediction
                               │
                               ▼
                      Survival evaluation
```

The cohort is 491 NPC (nasopharyngeal carcinoma) T1 MRI cases, 371 of which have usable outcome labels, evaluated with a Cox proportional-hazards objective and validated with **nested cross-validation + a label-permutation null**, not a single train/valid/test split (single-split numbers in this project have repeatedly turned out to be noise — see [Reproducibility notes](#reproducibility-notes)).

---

## TL;DR — where do I start reading?

This is a research repo, not a packaged library — there's no single `main.py`. The fastest way to understand *what was actually found*, without reading source code, is:

1. **`resume/linear_head/README.md`** — full narrative of the primary study (radiomics + DINOv3 → linear Cox head), task-by-task, with figures and the headline result (nested C-index **0.643** for image-based death prediction).
2. **`resume/transformer/README.md`** — the follow-up study reproducing the paper's hybrid ViT-Cox architecture (`2604.21056v1.pdf`) on the same cohort.
3. **`save/README.md`** — the original, less-curated lab-notebook version of the same results (chronological updates as they happened, including the bugs that were caught along the way).

If you want to *run* code instead, jump to [Running things](#running-things).

---

## Repository structure

```text
.
├── README.md                  ← you are here
├── 2604.21056v1.pdf            Reference paper this project reproduces/extends
├── .gitignore
│
├── config/                     All experiment configs (YAML), one per variant
│
├── data/
│   ├── labels/                 Per-case outcome labels + survival ground truth
│   ├── radiomics/               Per-case cached pyradiomics features
│   └── splits/                  Train/valid/test case-ID lists (random + stratified)
│
├── src/                         All source code, one folder per pipeline
│   ├── linear_model/             Radiomics + DINOv3 → linear Cox head (main study)
│   ├── linear_model_Merlin/      Same, with Merlin (3D CT) backbone instead of DINOv3
│   ├── linear_model_multislice/  Multi-slice aggregation variants
│   ├── ViT_cox/                  Hybrid ViT-Cox (paper architecture), DINOv3 image branch
│   ├── ViT_cox_Merlin/           Same hybrid architecture, Merlin image branch
│   ├── backbone_benchmark/       Sweep of alternative vision backbones (CLIP, SAM, etc.)
│   ├── feature_prunning/         Radiomics feature selection (per outcome)
│   └── test/                     Small performance/sanity scripts
│
├── output/, output_merlin/, output_bench_*/, output_multislice_*/
│                                 Raw run artifacts (metrics, plots, cross-val results)
│                                 — one directory per experiment variant
├── ViT_output/, ViT_output_merlin/, ViT_output_merlin_roi/
│                                 Raw run artifacts for the hybrid ViT-Cox models
│
├── save/                        Curated, per-experiment result folders + summaries
│                                 for the linear-head study (chronological)
│
└── resume/
    ├── linear_head/README.md     Narrative write-up: linear-head study, full story
    └── transformer/README.md     Narrative write-up: hybrid ViT-Cox study, full story
```

A few naming conventions worth knowing up front, since they explain most of the directory sprawl:

- **`output*` / `ViT_output*` directories are one-per-experiment-variant.** `output_bench_clip/`, `output_bench_sam1/`, `output_multislice_top5/`, etc. are all raw results from the same underlying pipeline, run with a different config. They're regenerated by the scripts in `src/`, not hand-maintained.
- **`save/` holds the linear-head study's results in the same "one folder per experiment" style, but curated** — each subfolder (`nested_cv_event/`, `confound_check_event/`, `finetune_event/`, …) has its own `summary.md`.
- **`resume/` is the readable layer on top of `save/` and `output*/`** — prose narratives that reference the raw folders instead of duplicating the numbers.

---

## Data

- **Cohort**: 491 NPC (nasopharyngeal carcinoma) patients, T1-weighted MRI with tumor segmentation masks. 371 of the 491 have usable outcome labels; the other 120 are imaged but unlabeled.
- **`data/labels/Case###.json`** — per-case outcome labels. Three outcomes are tracked independently, each with its own event flag *and* its own time-to-event column (these are **not interchangeable** — see `config/config.yaml`'s `event_time_columns` map):
  - `event` / `time_months` (death) — 71 events, the best-powered outcome.
  - `metastasis` / `metastasis_t_months` — 40 events.
  - `recurrence` / `recurrence_t_months` — 23 events, likely underpowered regardless of modeling choices.
- **`data/radiomics/Case###.json`** — 107 pyradiomics features per case (shape, first-order, texture), extracted by `data/radiomics/make_radiomics.py`.
- **`data/splits/`** — `stratified_{train,valid,test}.csv` (260/56/55 cases, stratified by the `event` label) is the default split used throughout; `complete_list.csv` (all 371 labeled cases) is used for nested cross-validation.
- External pretraining corpora (LUNA16 lung-CT, a domain-matched NPC-MRI dataset) are referenced but not vendored — see `ViT_output/data/*/download_links.txt`.

---

## Prediction targets

| Outcome | Column pair | Total events (of 371) | Status |
|---|---|---|---|
| Death | `event` / `time_months` | 71 | Best-powered; the one validated result in the project |
| Metastasis | `metastasis` / `metastasis_t_months` | 40 | Plausible lead, not significant (p≈0.10) |
| Recurrence | `recurrence` / `recurrence_t_months` | 23 | Data-scarcity floor; not usable |

---

## Two model families

**1. Linear head (`src/linear_model*/`)** — a frozen vision backbone produces one embedding per case, cached to disk, and a single linear layer maps that (plus/minus radiomics) to a Cox risk score. Simple, cheap, and — per the validated results — currently the **best-performing** approach in this repo.

**2. Hybrid ViT-Cox (`src/ViT_cox*/`)** — reproduces the two-branch architecture from the reference paper (`2604.21056v1.pdf`, Fig. 4): an image branch plus a radiomics-patch transformer branch, aligned by a contrastive loss and mixed into one Cox risk. More complex, and — per the validated results — has not yet beaten the linear head on this cohort.

Both families share the same underlying data, splits, and evaluation code style (`nested_cv.py`, `baseline.py` with a label-permutation null), and both have a DINOv3 variant and a Merlin (3D CT foundation model) variant. `src/linear_model_multislice/` and `src/backbone_benchmark/` are extensions of family 1: testing whether feeding more image slices, or swapping the backbone, changes the result.

---

## Config files

Everything experiment-specific (backbone, input mode, hyperparameters, cache paths) lives in `config/*.yaml`, not hardcoded in scripts — scripts take `--config path/to/file.yaml`. `config/config.yaml` is the base config for the linear-head/DINOv3 pipeline; every other file is a variant of it for a specific experiment:

| Config | What it changes |
|---|---|
| `config.yaml` | Base: DINOv3 backbone, linear head |
| `config_merlin*.yaml` | Merlin 3D CT backbone (whole-volume / ROI-cropped variants) |
| `config_bench_*.yaml` | One per alternative backbone in the benchmark sweep (CLIP, SAM 1/2, Rad-DINO, PubMedCLIP, CT-FM, DINOv3+CLIP fusion) |
| `config_multislice_*.yaml` | Multi-slice aggregation variants (top-k / all-slices) |
| `vit_cox*.yaml` | Hybrid ViT-Cox architecture configs (DINOv3 / Merlin image branch, ROI variants) |

---

## Outputs — how to tell them apart

- **`output/`, `ViT_output/`** — the *default*-config runs (DINOv3, linear head and hybrid respectively).
- **`output_<variant>/`, `ViT_output_<variant>/`** — same pipeline, run with the matching `config_<variant>.yaml`. The suffix always matches a config file name.
- Inside each: `metrics_*.json` / `nested_cv_*.json` (numeric results), `plots/` (loss curves, Kaplan-Meier plots, null-distribution histograms), and sometimes `cache/` (cached feature embeddings — usually gitignored, see `.gitignore`'s `*.npz` rule).
- **`save/`** duplicates some of this for the linear-head study specifically, but each experiment folder also has a `summary.md` explaining the result in prose — check there first before parsing raw JSON.

---

## Recommended way to navigate the code

1. **Read `resume/linear_head/README.md` first.** It walks through the whole pipeline task-by-task (radiomics extraction → feature pruning → caching → training → baseline nulls → nested CV → confound checks → interpretability → backbone swaps) and links each step to the exact script and config that produced it.
2. **Cross-reference with `src/linear_model/`** while reading — the file names match the tasks almost 1:1: `caching_features.py`, `dataset.py`, `head_model.py`, `training.py`, `baseline.py`, `nested_cv.py`, `confound_check.py`, `finetune.py`, `interpretability.py`.
3. **Then `resume/transformer/README.md`** for the more complex hybrid architecture, cross-referenced with `src/ViT_cox/`.
4. **Variant folders** (`_Merlin`, `_multislice`, `backbone_benchmark`) are best understood as "the same pipeline, one thing changed" — each has its own short README or is explained in a dedicated section of the two `resume/` narratives.
5. Anything you can't find in the narrative write-ups is almost certainly explained in the corresponding `save/<experiment>/summary.md` or in a comment at the top of the relevant config file.

---

## Running things

There's no single top-level entry point; each pipeline is run from its own `src/` subfolder against a config. The general shape (using the main linear-head/DINOv3 pipeline as the example):

```bash
cd src/linear_model

# 1. Extract radiomics features (once), then prune per outcome.
python ../../data/radiomics/make_radiomics.py
python ../feature_prunning/prune_features.py

# 2. Cache image (and radiomics) features for the head to train on.
python caching_features.py

# 3. Train + evaluate a single split.
python training.py --input-mode image      # or radiomics / both

# 4. Get an honest, higher-power result: label-permutation null + nested CV.
python baseline.py
python nested_cv.py
```

Add `--event-type {event,recurrence,metastasis}` to any script to switch outcomes. The Merlin variants (`src/linear_model_Merlin/`, `src/ViT_cox_Merlin/`) require a separate `MERLIN` conda environment — see `src/linear_model_Merlin/README.md` for the exact setup and a caveat about proxy settings breaking the HuggingFace download.

---

## Headline results

The one number in this repository that's held up under every re-check (seed stability, confound analysis, bug audits): **predicting death (`event`) from a single tumor-cropped DINOv3 image embedding, via one linear Cox layer, scores a nested cross-validated C-index of 0.643 (p=0.000)** — the best result across every architecture and backbone tried, including the more complex hybrid ViT-Cox model and every alternative vision backbone benchmarked (CLIP, Merlin, SAM, Rad-DINO, PubMedCLIP, CT-FM).

Full comparison tables and the reasoning behind them live in `resume/linear_head/README.md` and `resume/transformer/README.md`.

---

## Reproducibility notes

- Throughout this project, **single train/valid/test split results have repeatedly turned out to be noise** once checked against nested cross-validation and a label-permutation null. Treat any number reported without a nested-CV p-value with caution.
- A handful of real bugs were caught and fixed mid-study (an event/time-column mismatch, a float32 precision artifact, a stale `img_size` config value) — see the `resume/` write-ups for what they were and how they were caught, since the same classes of bug are easy to reintroduce when adding a new variant.
- No lockbox test set currently exists for either study; every reported number is internal nested-CV validation on the same 371-case cohort.

---

## Reference

This project reproduces and extends the architecture described in the accompanying paper, `2604.21056v1.pdf` — *Radiomics-Guided Vision Transformers for Survival Analysis* (see `resume/transformer/README.md` for exactly which parts were reproduced verbatim vs. modified).
