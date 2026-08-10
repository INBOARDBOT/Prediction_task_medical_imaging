# Contrastive-loss temperature fix -- nested-CV validation

## Why this run exists

The lambda_loss sweep (`save/nested_cv_event/` sibling work, see
`resume/transformer/README.md` Task 4) diagnosed the NT-Xent contrastive
loss as stuck exactly at `ln(260)=5.561`, the loss a 260-way classifier
gets from pure random guessing -- it never learned any cross-modal
alignment, at any lambda weight. Hypothesis: `temperature=0.5` was too
high, washing out weak similarity structure into a near-uniform softmax.

## Method

Swept temperature in {0.5, 0.2, 0.1, 0.07, 0.03, 0.01} on the fast
single-split check, using the NPC-MRI-pretrained radiomics encoder
(the checkpoint most responsive to this fix -- the same sweep on the
LUNA16-pretrained checkpoint showed almost no effect, see below).
`temperature=0.1` gave both the best single-split test C-index (0.682)
and, critically, a genuinely decreasing contrastive loss during training
(5.66 -> 5.27 -> 4.36 over 40 epochs, confirmed well below the random-
guessing floor) -- unlike every previous configuration, this one is
actually learning cross-modal alignment. Validated with the full
nested-CV protocol (5x5-fold, 30-permutation null) used throughout.

## Result

| pretrain corpus | temperature | contrastive loss learns? | single-split test C-index | nested-CV C-index | null mean +/- std | empirical p |
|---|---|---|---|---|---|---|
| LUNA16 | 0.5 (default) | no (stuck at 5.56) | 0.588 | 0.609 | 0.505 +/- 0.026 | 0.000 |
| NPC MRI | 0.5 (default) | no (stuck at 5.56) | 0.588-0.632 | 0.573 | 0.497 +/- 0.039 | 0.000 |
| NPC MRI | 0.1 (fixed) | **yes** (5.66->4.36) | **0.682** | **0.565** | 0.503 +/- 0.039 | **0.033** |

## Interpretation

**This is the fourth time this session a single-split "improvement" has
evaporated under nested CV** (after the metastasis event/time-column bug,
the both-mode single-split jump, and the NPC-MRI-pretrain-vs-LUNA16
comparison). The temperature fix genuinely works as a fix -- the
contrastive loss really does learn cross-modal structure now, confirmed
by its trajectory, not just inferred from a downstream metric. But making
the contrastive objective actually succeed did not translate into better
survival prediction; if anything, this is the lowest nested-CV C-index of
any hybrid configuration tried, and the p-value (0.033) is far weaker and
closer to the null band than every other significant result in this
project (all others were p=0.000, comfortably clear of their null; this
one sits right at the null distribution's edge, with one of 30 null runs
actually exceeding it).

Plausible explanation: forcing the two branches into an aligned
representation space may pull each branch's embedding toward whatever
correlational structure exists between global image appearance and
patch-radiomics texture -- structure that isn't necessarily the same as
"structure useful for predicting survival." A successfully-learned
contrastive objective can still be optimizing the wrong thing for this
specific downstream task, even though it's no longer stuck at chance.

## Bottom line

"Fixing" the contrastive loss mechanically succeeded but did not improve
(and arguably hurt) the validated result. The plain linear head on the
frozen DINOv3 CLS token (0.643) remains the best-validated model in this
entire project by a clear margin over every hybrid variant tried,
including this one.
