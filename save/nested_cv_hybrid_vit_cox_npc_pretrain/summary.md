# Hybrid ViT-Cox nested-CV, NPC-MRI pretraining vs. LUNA16 pretraining

## Why this run exists

Swapped the radiomics transformer's pretraining corpus from LUNA16 (lung
CT nodules -- anatomically and modality-mismatched to our task) to a
domain-matched external dataset: 277 primary NPC patients, T1-weighted
MRI with tumor segmentation masks (Zenodo 10.5281/zenodo.13131827) --
same disease, same anatomy, same sequence as our own cohort. Reused the
exact tumor-mask ROI-crop logic already validated on our own data
(`prepare_npc_mri_pretrain.py`), instead of the nodule-center
approximation LUNA16 needed.

276/277 patients (99.6%) yielded a usable crop. Notably, only **16
features** survived the zero-variance filter on this corpus (vs. 93 for
LUNA16) -- and 16 is exactly what our own NPC cohort's radiomics tokens
independently reduce to, a good sign the domain match is real at the
feature level, not just thematic.

## Method

Identical nested-CV protocol to `save/nested_cv_hybrid_vit_cox/` (LUNA16
version): 5x5-fold stratified outer CV over all 371 cases, 30-permutation
null. Only the radiomics encoder's pretraining initialization changed.

## Result

| pretraining corpus | crops | features kept | single-split test C-index (smoke test) | nested-CV C-index | null mean +/- std | empirical p |
|---|---|---|---|---|---|---|
| LUNA16 (lung CT nodules) | 368 | 93 | 0.588 | **0.609** | 0.505 +/- 0.026 | 0.000 |
| **NPC MRI (domain-matched)** | 276 | 16 | **0.632** | **0.573** | 0.497 +/- 0.039 | 0.000 |

## Interpretation

**The domain-matched pretraining looked like a clear win on the
single-split smoke test (0.632 vs 0.588) but performed *worse* under the
rigorous nested-CV check (0.573 vs 0.609).** Both remain statistically
significant against their own null, so this isn't "domain match doesn't
work" -- it's the same lesson this whole session keeps surfacing: a
single train/valid/test split is not a reliable signal for comparing two
configurations, even when the difference looks large and intuitive.

Plausible reasons NPC-MRI pretraining scored lower despite the better
domain match:
- **Fewer features to learn from**: 16 vs 93 kept dimensions gives the
  masked-token reconstruction task much less to work with -- less
  pretraining signal, even if the 16 that exist are more relevant.
- **Fewer pretraining examples**: 276 vs 368 crops.
- **The gap is small relative to null variance**: nested-CV null std here
  (0.039) is wider than LUNA16's (0.026) -- both observed values clear
  their own null band, but the two configurations' bands overlap
  partially, so "0.609 > 0.573" is a real difference in this run but not
  necessarily a large, robust one without a direct head-to-head
  significance test (not performed here).

## Bottom line

Neither pretraining source beats plain `image`-only from the linear-head
study (0.643). Domain-matched pretraining is intuitively the right idea
and is retained as the default going forward (matches the actual
downstream task, more reusable if the NPC cohort grows), but the LUNA16
result is a reminder not to assume "more domain-appropriate" trivially
means "better nested-CV number" without checking.
