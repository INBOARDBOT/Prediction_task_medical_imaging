# Nested CV (full 371-case cohort) -- event_type = metastasis

## Method

Same procedure as `nested_cv_event/`: 5 repeats x 5-fold stratified outer
CV over all 371 cases, aggregated out-of-fold concordance index, 100
label-permutation null passes. Uses the correct `metastasis_t_months` time
column.

## Results

| input_mode | aggregated C-index (371 cases) | null mean ± std | empirical p |
|---|---|---|---|
| radiomics | 0.486 | 0.498 ± 0.048 | 0.610 |
| image | 0.573 | 0.498 ± 0.054 | 0.100 |
| both | 0.536 | 0.505 ± 0.055 | 0.290 |

40 total metastasis events across the full cohort (vs. 23 for recurrence,
and 71 for `event`/death) -- more events than recurrence, fewer than
death.

## Interpretation

No result clears conventional significance, but `image` mode (0.573,
p=0.100) is the closest thing to a lead here -- similar shape to the
`event` result (image beating radiomics and both) but far weaker evidence:
p=0.10 vs. p=0.000 for `event`. With 40 events this cohort has more power
than recurrence (23 events) but still meaningfully less than `event`
(71 events), which likely explains the gap: same qualitative pattern
(image > both > radiomics), lower statistical confidence.

## Bottom line

Not significant, but not as clearly "nothing here" as radiomics-only or
recurrence. Worth revisiting if the cohort grows -- the effect may be real
at the same magnitude as `event`'s but this dataset doesn't have enough
metastasis events yet to confirm it the way it could for death.
