# Baseline calibration results

**Update:** the single-split baseline check below found nothing
significant for any outcome. Follow-up work reran all three outcomes with
a much higher-power nested-CV procedure over the full 371-case cohort
(`src/linear_model/nested_cv.py`, 5x5-fold, aggregated out-of-fold
concordance index, 100-permutation null) instead of one fixed 55-case test
split:

| event_type | total events (371 cases) | best mode | nested C-index | empirical p |
|---|---|---|---|---|
| event (death) | 71 | **image** | **0.643** | **0.000** |
| metastasis | 40 | image | 0.573 | 0.100 |
| recurrence | 23 | radiomics | 0.563 | 0.200 |

Only `event_type=event` clears significance, and decisively so (`image`
and `both` both p=0.000, beyond the max of 100 null runs; see
`nested_cv_event/summary.md`). `metastasis` shows the same qualitative
pattern (image > both > radiomics) but far weaker evidence (p=0.10),
plausibly because it has fewer events than `event` but more than
`recurrence` (see `nested_cv_metastasis/summary.md`). `recurrence` shows
nothing under either evaluation scheme -- with only 23 total events in the
whole cohort, this looks like a genuine data-scarcity limit that more
powerful evaluation can't work around (see
`nested_cv_recurrence/summary.md`). `radiomics`-only shows no signal for
any outcome under either scheme.

The single-split result for `event` wasn't wrong, it was underpowered --
worth keeping in mind before writing off a modest/borderline result as
"no effect" when the test set is this small.

**Second update:** before trusting the `event`/`image` result as real
tumor biology, `confound_check_event/` checked it against plausible
technical shortcuts (crop size, slice position, intensity calibration,
tumor volume). It survives: no slice-position artifact, weaker volume
correlation than the non-predictive `radiomics` baseline (which argues
against "it's just size in disguise"), and it stays significant
(p=0.0006) in a bivariate Cox model even after directly adjusting for
tumor volume, whose own effect disappears once image risk is included.
One mild flag noted (intensity-correlation sign differs between image and
radiomics risk) but no acquisition/scanner metadata exists to test that
further. See `confound_check_event/summary.md`.

**Third update:** tried to fix `both` mode underperforming `image` alone
(0.624 vs 0.643 nested C-index) by replacing naive feature concatenation
with `TwoBranchCoxHead` -- separate linear projections per modality summed
at the output, with `radiomics` regularized much harder than `image`
(`src/linear_model/head_model.py`, `training.build_model_and_optimizer`).
This looked like a large win on the single train/valid/test split (test
C-index 0.496 -> 0.629), but re-checked under the same nested-CV procedure
used everywhere else here, the real number barely moved: 0.624 -> 0.617,
still below `image` alone and still p=0.000. The single-split jump was
mostly noise. `image`-only remains the best input_mode for `event`;
`radiomics` has nothing to add here under any combination scheme tried so
far. See `nested_cv_event/summary.md` (updated results table).

**Fourth update -- three follow-ups on the flagship `event`/`image` result:**
- `seed_stability_event/`: reran the nested CV across 5 different fold
  seeds. Result holds up (mean 0.622 +/- 0.014), unlike the two prior false
  leads above -- this one isn't fold-assignment luck.
- `interpretability_event/`: patch-level heatmaps show the signal is
  diffuse across the whole tumor crop, not localized to a sub-region --
  more a texture/appearance signature than a spotted feature.
- `finetune_event/`: tried unfreezing the last DINOv3 block. Looked like a
  win (0.588 -> 0.666 mean test C-index) until a frozen-@-224 control
  (same code, nothing trainable in the backbone) showed the entire gain
  was a resolution effect (768px -> 224px), not fine-tuning -- the control
  and the fine-tuned run are statistically indistinguishable (0.670 vs
  0.666). Unplanned finding: `img_size=768` used throughout this whole
  study may be suboptimal; re-checking `img_size=224` under full nested CV
  is now the highest-value next step, ahead of further fine-tuning.

A curated, narrative version of the full study (this file plus all of the
above) lives in `resume/linear_head/README.md`.

Method (`src/linear_model/baseline.py`): for each outcome (`event_type`) and
each `input_mode` (radiomics / image / both), the exact same
train -> early-stop-on-valid -> test procedure used in `training.py` is
rerun 200 times with the (event, time) labels shuffled independently
within each split (train/valid/test each permuted among themselves, so
split sizes and censoring rates stay identical to the real experiment, but
any true feature-outcome association is destroyed). This gives an
empirical null distribution of test concordance index the pipeline would
produce on pure noise at this sample size. `empirical_p_value` = fraction
of null runs that scored at least as high as the real (unshuffled) run --
i.e. how often chance alone beats what we observed.

Also included: a classical single-covariate CoxPH (lifelines) fit on tumor
volume alone (`original_shape_MeshVolume`), as the simplest clinically
meaningful floor.

## A bug was caught and fixed mid-analysis

The first pass of this analysis (recurrence, metastasis) paired each
event_type's event indicator with the wrong time column: the code always
used `time_months` (time to death/overall-survival) instead of the
matching `recurrence_t_months` / `metastasis_t_months`. These differ in
30/371 cases (e.g. a patient who died at 51 months but whose recorded
recurrence follow-up extends to 63 months) -- so recurrence/metastasis
models were being fit against death timing, not their own event timing.
This was fixed in `src/linear_model/dataset.py` and `baseline.py` (config
now has an explicit `data.event_time_columns` map). All numbers below are
from the corrected code. Concretely, fixing this changed the headline
result: the one nominally "significant" finding from the first pass
(metastasis + image, p=0.050) was an artifact of the bug -- it drops to
p=0.37 once the correct time column is used. `event_type=event` was never
affected (its own correct time column already was `time_months`).

## Results (corrected)

| event_type | train events | test events | mode | observed test C-index | null mean ± std | empirical p |
|---|---|---|---|---|---|---|
| event | 50/260 | 10/55 | radiomics | 0.591 | 0.509 ± 0.096 | 0.195 |
| event | 50/260 | 10/55 | image | 0.599 | 0.498 ± 0.109 | 0.180 |
| event | 50/260 | 10/55 | both | 0.496 | 0.502 ± 0.103 | 0.525 |
| recurrence | 16/260 | 2/55 | radiomics | 0.140 | 0.498 ± 0.209 | 0.965 |
| recurrence | 16/260 | 2/55 | image | 0.796 | 0.535 ± 0.220 | 0.150 |
| recurrence | 16/260 | 2/55 | both | 0.484 | 0.513 ± 0.204 | 0.560 |
| metastasis | 26/260 | 8/55 | radiomics | 0.574 | 0.514 ± 0.109 | 0.285 |
| metastasis | 26/260 | 8/55 | image | 0.566 | 0.524 ± 0.113 | 0.370 |
| metastasis | 26/260 | 8/55 | both | 0.557 | 0.502 ± 0.116 | 0.310 |

Tumor-volume-only classical CoxPH baseline: event 0.570, recurrence 0.280,
metastasis 0.516 test C-index (this baseline is correctly event_type-aware
too -- it also had the same time-column bug, now fixed, which is why its
recurrence number changed from an earlier, incorrect 0.570 to 0.280).

## Interpretation

- **Nothing here clears the noise floor at conventional significance.**
  Every empirical p-value is well above 0.05. The pipeline is behaving
  correctly (loss curves, early stopping, and the null distributions all
  look exactly as they should) -- there just isn't validated signal yet in
  any input_mode for any outcome at this sample size.
- **`event` (death)**: best result (image, 0.599) beaten by chance alone
  18% of the time. Not usable as-is.
- **`recurrence`**: still unreliable as a metric, not just a modeling
  question -- only 2 events in the 55-case test set, so concordance index
  is a near-degenerate statistic decided by a couple of pairwise
  comparisons. The 0.796 `image` result looks eye-catching but its null
  band is enormous (std ~0.22) for the same reason, and p=0.15 confirms
  it's not distinguishable from that noise. The 0.140 radiomics result is
  equally not meaningful, just unlucky on 2 events. Don't trust anything
  from this row without a larger test set or more recurrence events.
- **`metastasis`**: this is where the bug mattered most -- before the fix,
  `image` mode looked like the one real lead (p=0.050). After pairing it
  with the correct `metastasis_t_months`, that result is gone (p=0.37,
  observed C-index actually dropped from 0.690 to 0.566). This is a good
  illustration of why the permutation-null check matters: it would have
  caught this as "not significant" even without knowing about the
  underlying bug, and re-running after the fix confirms there was nothing
  there.
- **Tumor-volume-only baseline**: 0.570 (event) and 0.516 (metastasis) sit
  in the same range as the fancier pipeline's results for those outcomes --
  the added machinery isn't earning its complexity yet. The recurrence
  volume baseline (0.280) is unreliable for the same 2-event reason as
  above.

## Bottom line

None of radiomics / image / both currently show validated prognostic
signal for any of the three outcomes tested, once measured against a
proper label-permutation null. This is a data/power problem (260 train /
55 test cases, 8-50 events depending on outcome), not a pipeline bug --
except for the one real bug found and fixed above, which is worth keeping
in mind as a reminder to always sanity-check event/time column pairings
per outcome before trusting a Cox result.

Per-event-type detail and plots: `baseline_event/`, `baseline_recurrence/`,
`baseline_metastasis/` (each has `baseline_<event_type>.json` and the three
null-distribution histograms, all regenerated after the fix).

---

**Fifth update -- the hybrid ViT-Cox reproduction of the origin paper's
architecture** (image branch swapped for DINOv3, radiomics branch =
shallow transformer pretrained on an external corpus then fine-tuned):
validated at 0.609 (LUNA16 lung-CT pretrain, alpha=0.5), 0.573
(domain-matched NPC-MRI pretrain), and 0.565/p=0.033 (NPC-MRI pretrain + a
diagnosed/fixed contrastive-loss temperature bug) -- all real signal,
none beating the linear head's 0.643. Full narrative, including the
disk-crisis LUNA16/NPC downloads, the float32 precision bug, the
flat-contrastive-loss diagnosis, and four separate instances of
single-split results not surviving nested CV: `resume/transformer/README.md`.
Per-run detail: `nested_cv_hybrid_vit_cox/`,
`nested_cv_hybrid_vit_cox_npc_pretrain/`, `nested_cv_hybrid_vit_cox_temp_fix/`.

**Sixth update -- mixed-risk alpha sweep**: only alpha=0.5 (paper default)
had been tried; radiomics has shown no independent signal in every check
this session, so swept alpha down toward the image branch. Result: a
clean, near-monotonic single-split trend (0.368 at alpha=1.0 up to 0.608
at alpha=0.0), and nested-CV validation of the extreme (alpha=0.0, LUNA16
pretrain) gave **0.612, p=0.000** -- the best hybrid-model result of the
whole project, though still below the linear head's 0.643. See
`nested_cv_hybrid_vit_cox_alpha0/summary.md`.

**Seventh update -- config/cache bug found while revisiting the
img_size=768-vs-224 lead flagged in `resume/linear_head/README.md`
Task 10.** `config.yaml`'s `image.img_size` had drifted to 768 via later
edits, but `caching_features.py` was never rerun after those edits -- the
cached DINOv3 embeddings behind *every* validated number in this entire
session (both studies, since the hybrid model reuses this same cache)
had actually been img_size=224 the whole time, not 768 as the config
suggested. Verified directly: a freshly regenerated genuine 768px cache
scores 0.475 (worse than chance) on the standard single-split check,
while regenerating at 224 exactly reproduces the familiar 0.588. So the
earlier "resolution explains the whole fine-tuning gap" framing was
comparing two secretly-identical resolutions through different code
paths, not testing 768 vs. 224 at all -- corrected in
`resume/linear_head/README.md`. Net result: there was no unrealized gain
to capture. 224 was already in use throughout, and is confirmed (now
correctly) to be the better choice. `config.yaml` now correctly reads
224, matching what's actually cached; a genuine 768px cache is kept at
`output/cache/image_features_dinov3_vits16plus_768_genuine.npz` for
reference.
