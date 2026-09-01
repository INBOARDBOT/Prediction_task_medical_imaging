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
- Merlin-based experiments;
- nested cross-validation;
- model interpretation and confound analysis.

> **Research use only.**
> This repository is intended for research and experimentation. The models are not clinically validated and should not be used for medical decision-making.

---

## Table of contents

- [Overview](#overview)
- [Main idea](#main-idea)
- [Repository structure](#repository-structure)
- [Data](#data)
- [Prediction targets](#prediction-targets)
- [Main pipeline](#main-pipeline)
- [Image features](#image-features)
- [Radiomics features](#radiomics-features)
- [Multimodal model](#multimodal-model)
- [Survival modelling](#survival-modelling)
- [Cross-validation](#cross-validation)
- [Feature caching](#feature-caching)
- [Configuration](#configuration)
- [Experimental branches](#experimental-branches)
- [Outputs](#outputs)
- [Recommended way to navigate the code](#recommended-way-to-navigate-the-code)
- [Reproducibility](#reproducibility)
- [Important implementation details](#important-implementation-details)
- [Reference](#reference)

---

# Overview

The goal of this project is to use medical imaging to predict clinical outcomes, with particular emphasis on **time-to-event prediction**.

Rather than treating the problem as a simple binary classification task, the models use survival-analysis concepts so that both:

1. whether an event occurred; and
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
