# Nested CV (full 371-case cohort) -- event_type = event (death)

## Why this run exists

The `save/baseline_*` results used a single fixed 55-case test split. None
of the three input_modes cleared the label-permutation noise floor there
(all p > 0.15). But a 55-case test set is underpowered to detect a
real-but-modest effect -- "not significant" there could mean "no signal"
or just "not enough test cases to see it." This run resolves that
ambiguity by evaluating over the full 371-case cohort instead of one fixed
holdout.

## Method

`src/linear_model/nested_cv.py`, 5 repeats x 5-fold stratified-by-event
outer CV (25 outer folds total) over all 371 labeled cases
(`data/splits/complete_list.csv`). Each outer-train portion is further
split into inner-train/inner-valid (85/15, stratified) purely for early
stopping; the model never sees its outer-test fold during training. Every
case gets 5 out-of-fold risk predictions (one per repeat), averaged into
one aggregated risk score per case, and the concordance index is computed
over all 371 cases' aggregated risk at once -- using ~7x more evaluation
data than the single-split baseline check.

A label-permutation null (100 permutations, labels shuffled across the
full cohort, entire nested-CV procedure rerun each time) gives the noise
floor this procedure would produce with no real signal.

## Results

| input_mode | aggregated C-index (371 cases) | null mean ± std | empirical p |
|---|---|---|---|
| radiomics | 0.476 | 0.490 ± 0.045 | 0.620 |
| **image** | **0.643** | 0.508 ± 0.037 | **0.000** |
| both (naive concat) | 0.624 | 0.509 ± 0.036 | **0.000** |
| both (TwoBranchCoxHead, current) | 0.617 | 0.500 ± 0.039 | **0.000** |

`empirical p = 0.000` means the observed value exceeded the max of all 100
label-permuted null runs, not merely the 95th percentile.

## Interpretation

- **This reverses the earlier "nothing is significant" conclusion for
  `image` and `both`.** With the full cohort's statistical power, the
  DINOv3 image embedding (frozen backbone, single ROI-cropped best-tumor
  slice) shows a real, robust association with overall survival -- not
  explainable by chance at this cohort size. The single-split baseline
  check wasn't wrong, it was underpowered: 55 test cases (10 events) just
  couldn't resolve an effect this size at conventional significance, even
  though the 371-case nested view can.
- **`radiomics` alone still shows nothing** (p=0.62, observed C-index is
  actually below the null mean). The pruned 15-feature radiomics signature
  does not carry validated prognostic signal for this outcome, even under
  the higher-power test.
- **`both` architecture fix (see `save/README.md` for the "#2" writeup)
  didn't meaningfully move the needle.** Switching from naive
  concatenation to `TwoBranchCoxHead` (separate per-modality linear
  projections, radiomics regularized much harder than image) looked like a
  big win on the single train/valid/test split (test C-index 0.496 -> 0.629),
  but the higher-power nested-CV number barely changed: 0.624 -> 0.617,
  both still comfortably below `image` alone (0.643) and both still
  p=0.000. The single-split jump was mostly evaluation noise, not a real
  improvement -- the same lesson the baseline-vs-nested-CV comparison
  already taught us, now confirmed on this specific architecture change
  too. `image`-only remains the best input_mode for this outcome;
  `radiomics` doesn't have anything to contribute here regardless of how
  it's combined.

## Caveat

This is a genuine, well-powered result for `event_type=event`, but it's
still an in-sample nested-CV estimate on this exact cohort (no cases held
back from all of this analysis) -- there is no more untouched data left to
report an unbiased final number against. If this pipeline needs a number
to report as a true held-out result later, new cases (not used in any
tuning or nested-CV run here) would be needed. Treat 0.643 as strong
evidence the signal is real, not as the number to publish as final
performance.
