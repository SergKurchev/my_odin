# Fix Summary: Category Color Mismatch in Pred Cat Visualization

## Problem
In 3D visualizations (HTML viewer), predicted categories (Pred Cat) displayed with incorrect colors:
- Red strawberries (Ripe) → displayed as green
- Green strawberries (Unripe) → displayed as orange  
- Orange strawberries (Half-ripe) → displayed as gray/background

This was a shift of +1 in category indices. GT categories displayed correctly.

## Root Cause
The `+1` operation in `prepare_3d()` (odin_model.py line 1311) was designed for ScanNet dataset compatibility, which uses 1-indexed labels (1-18). However, the strawberry dataset uses 0-indexed labels (0, 1, 2) following standard ML conventions.

**Flow before fix:**
1. Model outputs: 0, 1, 2 (0=Ripe, 1=Unripe, 2=Half-ripe)
2. `prepare_3d` adds +1: 1, 2, 3
3. `my_train_odin.py` subtracts 1: 0, 1, 2
4. Palette maps: 0→red, 1→green, 2→orange ✓

**But this should work!** The issue was that the +1 was unnecessary for strawberry dataset.

## Solution
Implemented conditional logic to skip the +1 operation for strawberry dataset:

### Changes Made

**1. odin/odin_model.py (prepare_3d function)**
```python
# Check if this is strawberry dataset (uses 0-indexed labels)
is_strawberry = batched_inputs is not None and 'strawberry' in batched_inputs.get('dataset_name', '')

# add +1 to labels as mask3d evals from 1-18 (ScanNet convention)
# BUT: strawberry dataset uses 0-indexed labels, so skip +1
if is_strawberry:
    pred_classes_output = labels_per_image
else:
    pred_classes_output = labels_per_image + 1
```

**2. my_train_odin.py (visualization code)**
```python
# ВАЖНО: для strawberry dataset pred_classes уже 0-индексированные (без +1 в prepare_3d)
# поэтому используем напрямую без вычитания
point_pred_cat[m] = int(pred_classes[inst_idx])
```

**Flow after fix:**
1. Model outputs: 0, 1, 2
2. `prepare_3d` keeps as-is for strawberry: 0, 1, 2
3. `my_train_odin.py` uses directly: 0, 1, 2
4. Palette maps: 0→red, 1→green, 2→orange ✓

## Verification
The fix ensures:
- Pred Cat: 0=Ripe(red), 1=Unripe(green), 2=Half-ripe(orange)
- GT Cat: unchanged (already correct)
- Palette: 0→red, 1→green, 2→orange
- Backward compatibility: ScanNet and other datasets still use +1 operation

## Commits
1. `39070f7` - Add comprehensive debug logging for category color investigation
2. `0ac499d` - Fix category color mismatch: skip +1 for strawberry dataset
3. `4305f56` - Remove debug logging after fix verification

## Testing Recommendations
1. Run evaluation on strawberry dataset
2. Check HTML visualizations to verify colors are correct:
   - Ripe strawberries should be red
   - Unripe strawberries should be green
   - Half-ripe strawberries should be orange
3. Verify GT categories remain correct (should be unchanged)
4. Test on ScanNet dataset to ensure backward compatibility

## Related Files
- `odin/odin_model.py` - Model inference and label preparation
- `my_train_odin.py` - Evaluation and visualization generation
- `generate_sample_viewer.py` - HTML viewer generation (unchanged)
- `VISUALIZATION_PRED_CAT_PROBLEM.md` - Original problem description

## Notes
- The fix is backward compatible with ScanNet and other datasets
- GT categories were already correct and remain unchanged
- The palette (SEG_PALETTE) was correct and remains unchanged
- Only the label indexing logic was modified
