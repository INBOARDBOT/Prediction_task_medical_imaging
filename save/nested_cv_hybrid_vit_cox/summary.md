# Hybrid ViT-Cox nested-CV validation (paper architecture, event_type=event)

**Update**: the radiomics transformer's pretraining corpus was later
swapped from LUNA16 (this run) to a domain-matched external NPC MRI
dataset. Counter-intuitively, domain-matched pretraining scored *lower*
under nested CV (0.573) despite scoring higher on a single-split smoke
test -- see `save/nested_cv_hybrid_vit_cox_npc_pretrain/summary.md` for
the full comparison. This LUNA16 result (0.609) remains the better of the
two pretraining sources tried so far.

## Method

Same protocol as `save/nested_cv_event/` (linear-head study): 5 repeats x
5-fold stratified outer CV over the full 371-case cohort, inner
train/valid split per outer fold for early stopping, aggregated
out-of-fold risk -> one C-index over all 371 cases, label-permutation
null (30 permutations here vs. 100 for the linear head -- each hybrid
training run is far more expensive: full transformer + contrastive loss
vs. a single linear layer -- so this trades permutation resolution
(~0.033 vs ~0.01) for tractability; full run took ~52 minutes).

Model: DINOv3 CLS token (frozen, cached) + radiomics shallow transformer
(pretrained on 368 LUNA16 crops, fine-tuned every fold) + mixed risk
(alpha=0.5) + NT-Xent contrastive alignment (lambda=0.5) -- see
`resume/linear_head` conversation / code for full architecture detail.

**Caveat on scope**: this run used the same "all-371-cases-rotate-through-
test" nested-CV protocol as the rest of this session -- see the
methodology discussion around this result for what that does and doesn't
validate (real-signal-vs-noise: yes; a clean held-out number for one
deployable model: no). No lockbox test set was carved out before this run.

## Result

Observed aggregated C-index: **0.609**
Null (30 permutations): mean 0.505, std 0.026
Empirical p-value: **0.000** (observed beats every one of the 30 null runs)

## Comparison to the linear-head study (same event_type=event, same 371-case protocol)

| model | nested C-index | empirical p |
|---|---|---|
| linear head, image only | **0.643** | 0.000 |
| linear head, both (naive concat) | 0.624 | 0.000 |
| linear head, both (two-branch fix) | 0.617 | 0.000 |
| **hybrid ViT-Cox (this run)** | **0.609** | **0.000** |
| linear head, radiomics only | 0.476 | 0.620 (not significant) |

## Interpretation

The hybrid architecture (paper's Fig. 4, with DINOv3 swapped in for the
image ViT and a LUNA16-pretrained shallow transformer for the radiomics
branch) shows a real, significant signal -- comfortably outside its null
band. But it does not beat the much simpler linear-head approaches: it's
the lowest-scoring of the four significant results, ~0.03 below plain
`image`-only. The added machinery (radiomics tokenization, LUNA16
pretraining, shallow transformer, contrastive alignment) is not currently
earning its complexity over a single linear layer on the frozen DINOv3
CLS token -- consistent with the pattern already seen for `both` mode in
the linear-head study (adding radiomics, in any form tried so far, has
never beaten `image` alone for this outcome).

Plausible contributors, not yet investigated: the contrastive loss term
was nearly flat across training in the single-split smoke test (barely
moved from ~5.56 over 200 epochs), suggesting the cross-modal alignment
objective isn't learning much and may just be adding noise/regularization
pressure rather than a useful signal; the radiomics branch, even after
pretraining, is still built on a shallow (1-block) transformer over a
weak base signal (the same pruned-adjacent radiomics features that showed
no standalone signal in the linear-head study).
