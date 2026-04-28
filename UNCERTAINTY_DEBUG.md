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

---

## Problem: Wrong Colors in Visualization (Red→Green, Green→Orange, Orange→Background)

**Symptom**: Predicted categories show wrong colors - everything shifted by +1.

## Root Cause

**NUM_CLASSES mismatch!** The config file has `NUM_CLASSES: 18` (for ScanNet), but Strawberry dataset has only 3 classes.

In `odin_model.py:1456-1468`:
```python
labels = torch.arange(num_classes, device=self.device)  # [0, 1, ..., 17] for num_classes=18
# ...
result_3d = {
    "pred_classes": labels_per_image + 1,  # Adds +1 to make 1-indexed
}
```

**What happens:**
1. Model trained on 18 classes predicts class indices 0-17
2. ODIN adds +1 → outputs 1-18
3. Visualization subtracts -1 → back to 0-17
4. But Strawberry only has classes 0, 1, 2!
5. If model predicts class 1 (thinking it's class 0 after +1), visualization shows class 0 (correct by accident)
6. If model predicts class 2, visualization shows class 1 → **color shift!**

## Solution

**CRITICAL**: Override `NUM_CLASSES` in training command:

```python
train_cmd = [
    # ... other args ...
    'MODEL.SEM_SEG_HEAD.NUM_CLASSES', '3',  # ← MUST SET TO 3 FOR STRAWBERRY!
]
```

This ensures:
- `labels = torch.arange(3)` → [0, 1, 2]
- `labels_per_image + 1` → [1, 2, 3]
- Visualization `-1` → [0, 1, 2] ✓ Correct!

---

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
train_cmd = [
    # ... paths and basic args ...
    
    # CRITICAL: Set correct number of classes!
    'MODEL.SEM_SEG_HEAD.NUM_CLASSES', '3',  # Strawberry has 3 classes
    
    # Bayesian inference
    'MODEL.BAYESIAN_TYPE', 'swag',  # or 'mc_dropout' or 'none'
    'MODEL.BAYESIAN_SAMPLES', '10',  # Number of samples (must be > 1)
    'MODEL.BAYESIAN_INFERENCE_DURING_TRAINING', 'False',  # Faster eval
    
    # SWAG parameters
    'MODEL.SWAG.START_EPOCH', '5',
    'MODEL.SWAG.UPDATE_FREQ', '5',
    'MODEL.SWAG.MAX_MODELS', '10',
    'MODEL.SWAG.RANK', '20',
    'MODEL.SWAG.NO_COV_MAT', 'False',
]
```

## Expected Behavior

- **Training**: Deterministic forward pass (fast)
- **Eval during training** (with `BAYESIAN_INFERENCE_DURING_TRAINING=True`): Bayesian inference with multiple samples
- **Pure inference** (`--eval-only`): Bayesian inference with multiple samples

## Commits

- `b7fd042`: Add debug output for GT and pred category values
- `632d740`: Revert incorrect GT category indexing change
- `5d5ee59`: Fix Bayesian inference logic and add detailed uncertainty debugging
- Previous: Added uncertainty computation but it was never called
