# Fine-tuning the DINOv3 backbone -- event_type=event, image mode

## Why

The biggest untried lever identified after confirming the `image`-mode
signal was real: the DINOv3 backbone is fully frozen and off-domain
(ImageNet/web-pretrained, never seen a medical image). Unfroze the last
transformer block (~8% of backbone params, 2.37M/28.7M) and trained it
jointly with the linear head.

## Method

`src/linear_model/finetune.py`. Scoped down from the main study's rigor
for tractability: image mode only, single train/valid/test split (not the
full 25-outer-fold nested CV), `img_size=224` (DINOv3's native pretraining
resolution) instead of the 768 used for the main cached embeddings --
both far cheaper (196 vs 2304 patch tokens/image) and more principled for
fine-tuning. Last block + final norm unfrozen (lr=1e-5), head trained at
lr=1e-3, both with weight_decay=0.01, early-stopped on valid Cox loss.
Each run: ~50-60 epochs, under a minute on GPU.

## Results

Fine-tuning alone, 5 seeds: test C-index 0.650-0.677 (mean 0.666, std
0.009) -- clearly above the original frozen-@-768 baseline (0.588). Looked
like a real win. But `img_size` changed at the same time as "frozen vs
fine-tuned," so this doesn't isolate which change mattered. Added a
frozen-@-224 control (`--n-unfrozen-blocks 0`, same code path, zero
backbone params trainable) to separate the two effects:

![Fine-tune vs frozen control](finetune_vs_frozen_control.png)

| | seed 42 | seed 1 | seed 7 | seed 123 | seed 2024 | mean | std |
|---|---|---|---|---|---|---|---|
| frozen @ 224 (control) | 0.653 | 0.677 | 0.671 | 0.668 | 0.680 | 0.670 | 0.009 |
| fine-tuned last block @ 224 | 0.650 | 0.677 | 0.665 | 0.665 | 0.674 | 0.666 | 0.009 |
| frozen @ 768 (original baseline) | -- | -- | -- | -- | -- | 0.588 | -- |

## Interpretation

**Fine-tuning added nothing measurable.** The frozen-@-224 control (0.670)
and the fine-tuned result (0.666) are statistically indistinguishable --
fine-tuning is if anything very slightly *lower*, well within noise. The
entire ~0.08 gap versus the original 0.588 baseline is explained by the
resolution change alone, not by adapting the backbone's weights.

This is a genuinely useful negative result with an unplanned side-finding:
**`img_size=768` (the setting used for the main study's cached embeddings
and the validated 0.643 nested-CV result) may itself be a suboptimal
choice** -- 224 looks meaningfully better on this single-split check,
consistently across 5 seeds. That's worth a proper nested-CV check before
concluding anything, since this whole session has repeatedly shown
single-split numbers can mislead (the metastasis time-column bug, the
both-mode "improvement" that evaporated). Flagged as a follow-up, not
acted on here to keep this task's scope to what was asked (fine-tuning).

## Caveats

- Single split only, not nested-CV validated at the rigor applied to the
  main `event`/`image` result -- treat 0.666-0.670 as a promising signal
  to re-check, not a replacement for the validated 0.643.
- Only the last block was tried; unfreezing more blocks, a lower/higher
  backbone lr, or more epochs were not swept given time constraints.
- Both the fine-tuned and control conditions here still beat the original
  0.588 baseline, largely from the resolution change -- so switching the
  main pipeline to 224px is the more promising, and much cheaper, next
  step compared to continuing to pursue backbone fine-tuning.
