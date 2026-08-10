# Interpretability -- event_type=event, image mode

## Why

The confound check ruled out a few specific technical shortcuts (crop
size, slice position, intensity, volume) but didn't show *what* the model
is responding to. This is a qualitative follow-up: visualize which parts
of the tumor crop drive the risk score.

## Method

`src/linear_model/interpretability.py`. DINOv3's final-layer patch tokens
live in the same normalized embedding space as the CLS token the trained
linear head (`output/final_model_event_image.pt`, single-split model) was
fit on. Standardizing each patch token with the same train-fit mean/std
and projecting onto the trained weight vector gives a per-patch "risk
contribution" (no backprop through the frozen backbone needed) --
upsampled and overlaid on the crop. Examples selected by the model's own
predictions: top-3 highest-risk cases that actually died, top-3
lowest-risk cases censored with >30 months follow-up (i.e. cases the model
is most confident about in each direction).

**Important**: heatmaps use a single shared color scale across all 6
examples. Per-image normalization (each image contrast-stretched to its
own min/max) was tried first and made every case look equally "hot" in
its own local range, hiding whether high-risk crops are systematically
more positive overall -- exactly the kind of artifact this whole session
has been catching repeatedly, so worth flagging as a mistake corrected
in-place rather than silently.

## Result

![Risk heatmaps](risk_heatmaps.png)

| group | mean patch score |
|---|---|
| died, high risk (3 cases) | +0.998, +1.158, +0.831 |
| censored >30mo, low risk (3 cases) | -0.085, +0.261, +0.116 |

## Interpretation

The high-risk group is overwhelmingly warm-toned (dark red/maroon) across
nearly the whole crop; the low-risk group is much lighter and mixed, with
patches of blue. This is a real, consistent separation -- but it's
**diffuse across the whole ROI, not localized to a specific sub-region**
like a single nodule or margin. That's a meaningfully different claim than
"the model spots a particular lesion feature": it looks more like an
overall tissue-appearance or texture signature across the cropped region
than a spatially specific finding. Consistent with the confound check
(which found no single simple shortcut like size or position explains it)
but doesn't pin down *what* the texture difference represents biologically
-- that would need either a radiologist's read of these examples or a
larger, systematic version of this analysis (e.g. averaging heatmaps over
many cases, not just 6 hand-picked ones) to say anything stronger.

## Caveat

6 examples, hand-picked as the model's most confident predictions in each
direction -- illustrative, not a systematic validation. The model used is
the single-split checkpoint, not a nested-CV model (nested-CV models
aren't persisted to disk), so treat this as "what does a representative
trained image-mode head look at," not "what does the validated 0.643
result specifically look at."
