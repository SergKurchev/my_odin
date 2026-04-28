# Uncertainty Debugging Guide

## Problem: Uncertainty Always Zero

**Symptom**: `mean_uncertainty: 0.0` in evaluation metrics despite enabling Bayesian inference.

## Root Cause

Bayesian inference was **never actually running** due to incorrect logic in `odin_head.py`:

```python
# OLD (BROKEN):
use_bayesian_inference = (
    not self.training and      # False during training
    bayesian_type != "none" and
    num_samples > 1
)
if self.training:
    if not bayesian_during_training:
        use_bayesian_inference = False  # Already False!
```

**Issue**: During training eval (`self.training=True`), `use_bayesian_inference` was always `False` regardless of `BAYESIAN_INFERENCE_DURING_TRAINING` setting.

## Solution

Fixed logic to properly enable Bayesian inference during training when requested:

```python
# NEW (FIXED):
use_bayesian_inference = (
    bayesian_type != "none" and
    num_samples > 1 and
    (not self.training or bayesian_during_training)  # Enable if eval OR explicitly enabled during training
)
```

## Uncertainty Formulas

Based on **"What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"** (Kendall & Gal, NIPS 2017):

### 1. Predictive Entropy (Total Uncertainty)
```
H[y|x,D] = -∑_c p̄(y=c|x,D) log p̄(y=c|x,D)
```
where `p̄(y=c|x,D) = (1/T)∑_t p(y=c|x,θ_t)` is the averaged prediction across T samples.

**Interpretation**: Total uncertainty in the averaged prediction.

### 2. Expected Entropy (Aleatoric/Data Uncertainty)
```
E_θ[H[y|x,θ]] = (1/T)∑_t [-∑_c p(y=c|x,θ_t) log p(y=c|x,θ_t)]
```

**Interpretation**: Average uncertainty of individual predictions. Represents irreducible uncertainty due to noisy data.

### 3. Mutual Information (Epistemic/Model Uncertainty)
```
I[y,θ|x,D] = H[y|x,D] - E_θ[H[y|x,θ]]
```

**Interpretation**: Difference between total and aleatoric uncertainty. Represents model uncertainty that can be reduced with more training data.

## Debug Output

When running inference, you'll now see detailed debug output:

```
=== BAYESIAN INFERENCE CONFIG ===
self.training: True
bayesian_type: swag
num_samples: 10
bayesian_during_training: True
use_bayesian_inference: True
=== END CONFIG ===

=== UNCERTAINTY DEBUG ===
Number of samples: 10
Logits stack shape: torch.Size([10, 1, 100, 3])
Probs variance shape: torch.Size([1, 100, 3])
Probs variance stats: min=0.000123, max=0.045678, mean=0.012345
Predictive entropy stats: min=0.123456, max=1.234567, mean=0.567890
Mutual information stats: min=0.001234, max=0.123456, mean=0.045678
=== END DEBUG ===
```

### Key Indicators

1. **Probs variance > 0**: Samples are actually different (Bayesian inference is working)
2. **Probs variance ≈ 0**: All samples identical (deterministic mode or bug)
3. **Mutual information > 0**: Model has epistemic uncertainty
4. **Mutual information ≈ 0**: Model is very confident (or only 1 sample)

## Configuration

In `kaggle_my_train_odin.py`:

```python
'MODEL.BAYESIAN_TYPE', 'swag',  # or 'mc_dropout' or 'none'
'MODEL.BAYESIAN_SAMPLES', '10',  # Number of samples (must be > 1)
'MODEL.BAYESIAN_INFERENCE_DURING_TRAINING', 'True',  # Enable during training eval
```

## Expected Behavior

- **Training**: Deterministic forward pass (fast)
- **Eval during training** (with `BAYESIAN_INFERENCE_DURING_TRAINING=True`): Bayesian inference with multiple samples
- **Pure inference** (`--eval-only`): Bayesian inference with multiple samples

## Commits

- `5d5ee59`: Fix Bayesian inference logic and add detailed uncertainty debugging
- Previous: Added uncertainty computation but it was never called
