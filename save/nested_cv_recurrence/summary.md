# Nested CV (full 371-case cohort) -- event_type = recurrence

## Method

Same procedure as `nested_cv_event/`: 5 repeats x 5-fold stratified outer
CV over all 371 cases, aggregated out-of-fold concordance index, 100
label-permutation null passes. Uses the correct `recurrence_t_months` time
column (see `save/README.md` for the event/time pairing bug that was
found and fixed before any of this work).

## Results

| input_mode | aggregated C-index (371 cases) | null mean ± std | empirical p |
|---|---|---|---|
| radiomics | 0.563 | 0.501 ± 0.073 | 0.200 |
| image | 0.531 | 0.505 ± 0.072 | 0.350 |
| both | 0.556 | 0.502 ± 0.074 | 0.240 |

## Interpretation

Unlike `event_type=event`, moving to the full-cohort nested CV does **not**
turn up a significant result here for any input_mode -- all three p-values
are well above 0.05, and this is with the same power boost that revealed a
real signal for `event`. The reason is different this time: it's not that
the *test set* was too small, it's that the *whole cohort* only has 23
recurrence events total (16 in the old train split, plus a handful spread
across valid/test). Nested CV can't manufacture events that don't exist --
no amount of resampling recovers the statistical power a cohort with more
recurrence cases would have.

## Bottom line

`recurrence` genuinely looks underpowered at the *data* level, not just
the *evaluation* level. This is a case where "accept the data isn't
powerful enough" is the right call for now, unless more recurrence-labeled
cases become available.
