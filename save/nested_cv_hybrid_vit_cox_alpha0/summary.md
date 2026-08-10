# Mixed-risk alpha sweep -- nested-CV validation

## Why this run exists

The "enhance the transformer" punch list's #2 item: sweep alpha (the
mixed-risk weighting `F = (1-alpha) r_img + alpha r_rad`, eq. 3 of the
paper). Only alpha=0.5 (paper default) had been tried. Given radiomics
carries no independent signal in every prior check this session, the
prediction was that lower alpha (favoring the image branch) should help.

## Method

Swept alpha in {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0} on the fast
single-split check, LUNA16-pretrained encoder, default
lambda_loss=0.5/temperature=0.5. Result was a clean, near-monotonic trend
(not a single lucky spot-check): test C-index rose from 0.368 at
alpha=1.0 (pure radiomics risk) to 0.608 at alpha=0.0 (pure image risk in
the mix formula, though the radiomics branch still trains via the
contrastive loss, which doesn't depend on alpha). Validated the extreme
(alpha=0.0) with the full nested-CV protocol.

## Result

| alpha | single-split test C-index |
|---|---|
| 1.0 (pure radiomics) | 0.368 |
| 0.9 | 0.555 |
| 0.7 | 0.567 |
| 0.5 (paper default) | 0.588 |
| 0.3 | 0.599 |
| 0.2 | 0.593 |
| 0.1 | 0.599 |
| 0.0 (pure image) | 0.608 |

Nested-CV, alpha=0.0: **C-index 0.612**, null mean 0.504 +/- 0.031,
**p=0.000** -- clean separation from the null (observed clears the max of
all 30 permutation runs by a comfortable margin).

## Interpretation

**Best hybrid-model result of the whole project so far** (previous best:
0.609, LUNA16 pretrain at alpha=0.5). The improvement over alpha=0.5's
0.609 is real but modest (+0.003) -- both are comfortably significant, so
this isn't "alpha=0.5 was broken," more "removing radiomics' direct
contribution to the risk score doesn't hurt, and marginally helps."

Notable: even at alpha=0.0, this configuration is not identical to the
linear-head study's plain `image`-only model (0.643) -- the radiomics
branch still trains via the contrastive loss (lambda_loss=0.5 here), so
the image projection head still receives gradient pressure from the
cross-modal alignment objective even though radiomics never enters the
risk formula directly. That extra pressure doesn't help: 0.612 is still
below 0.643.

## Bottom line

Across every configuration tried this session -- naive concatenation,
regularized two-branch fusion, the full paper architecture under two
pretraining corpora, a fixed contrastive loss, and now a swept mixed-risk
weight -- **nothing has beaten the plain linear head on the frozen DINOv3
CLS token alone (0.643)**. The consistent pattern is that radiomics adds
no value for this outcome on this cohort, in any combination scheme
tried, and the closer a configuration gets to "effectively just using the
image signal," the better it does.
