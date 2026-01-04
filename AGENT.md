# Annotation Tool Bug Fixes - Progress Tracker

## Requirements Checklist

- [ ] Two types of prompts: Box Prompts and Point Prompts (text prompts excluded for now)
- [ ] Each prompt type should have positive and negative versions
  - [ ] Positive prompt creates new instance if none selected
  - [ ] Positive prompt expands/modifies existing instance if one is selected
  - [ ] Negative prompt refines existing instance (doesn't delete it) - backend rejects negative prompts without selection
- [ ] Single point click should only ever create or modify a single instance - backend handles this
- [ ] Segmentation mask should stay within its bounding box - implemented clip_mask_to_bbox function
- [ ] Two masks cannot overlap (bounding boxes may overlap, but not masks) - implemented remove_mask_overlap function
- [ ] Prevent creating instances by clicking on existing masks - backend checks and auto-selects instance
- [ ] When deleting an instance, delete all associated prompts (positive and negative) - updated remove_instance endpoint
- [ ] Negative prompts should refine masks, not remove them entirely - backend handles refinement correctly

## Bugs Identified

1. ✅ **Mask extends beyond bounding box**: Masks can have pixels outside their bounding boxes - FIXED
2. ✅ **Mask overlap allowed**: Can create new instances by clicking on existing masks - FIXED (auto-detects and refines instead)
3. ✅ **Masks can overlap**: Two masks can have overlapping pixels - FIXED (overlap removed when creating/refining instances)
4. **Prompts not tracked per instance**: Current system only tracks one prompt per instance, not all prompts
5. **Negative prompts reset instead of refine**: Negative prompts don't accumulate with existing prompts
6. **Instance deletion incomplete**: Only removes one prompt, not all prompts associated with an instance
7. **Single click can affect multiple instances**: Need to ensure only one instance is affected per click

## Implementation Plan

### Phase 1: Prompt Tracking System

- [ ] Create `instance_prompts` data structure to track all prompts per instance
- [ ] Track both positive and negative prompts (boxes and points) per instance
- [ ] Update prompt addition logic to append to instance's prompt list

### Phase 2: Mask Constraints

- [ ] Clip masks to bounding boxes after segmentation
- [ ] Check for mask overlap before creating new instances
- [ ] Prevent point clicks on existing masks

### Phase 3: Refinement Logic

- [ ] Accumulate prompts for refinement (don't reset)
- [ ] Ensure negative prompts refine instead of delete
- [ ] Handle multiple prompts per instance correctly

### Phase 4: Instance Deletion

- [ ] Delete all prompts associated with an instance
- [ ] Update prompt indices for remaining instances

### Phase 5: Frontend Updates

- [ ] Check if click is on existing mask before creating new instance
- [ ] Update UI to show all prompts per instance
- [ ] Handle instance selection properly

## Implementation Summary

### Completed Changes

1. **Prompt Tracking System**

   - Created `instance_prompts` data structure to track all prompts (positive/negative, box/point) per instance
   - Each instance now has a list of all prompts associated with it
   - Updated both box and point prompt handlers to track prompts properly
2. **Mask Constraints**

   - Implemented `clip_mask_to_bbox()` function to ensure masks stay within their bounding boxes
   - Masks are clipped during serialization and after refinement
   - Implemented `remove_mask_overlap()` function to prevent mask overlap
   - New instances automatically have overlap removed with existing masks
3. **Click Detection**

   - Implemented `check_point_on_mask()` function to detect if a point click is on an existing mask
   - For positive prompts: automatically selects the instance if clicking on its mask
   - For negative prompts: requires instance selection (backend rejects if none selected)
4. **Refinement Logic**

   - Updated refinement handlers to clip masks to bounding boxes
   - Updated refinement handlers to remove overlap with other masks
   - Negative prompts now properly refine instances instead of deleting them
5. **Instance Deletion**

   - Updated `remove_instance` endpoint to delete ALL prompts associated with an instance
   - Properly updates prompt indices for remaining instances
   - Removes prompts from both `prompted_boxes` and `prompted_points` lists

### Key Functions Added

- `clip_mask_to_bbox(mask, bbox)`: Clips a mask to its bounding box
- `check_point_on_mask(point, masks, boxes, img_w, img_h)`: Checks if a point is on any existing mask
- `remove_mask_overlap(new_mask, existing_masks, new_box)`: Removes overlap between new mask and existing masks

### Files Modified

- `app/backend/main.py`:
  - Added helper functions for mask clipping and overlap removal
  - Updated `serialize_state()` to clip masks to bounding boxes
  - Updated `add_box_prompt()` and `add_point_prompt()` endpoints
  - Updated `remove_instance()` endpoint to delete all prompts

## Testing Notes

- Test positive point clicks on existing masks (should auto-select instance)
- Test negative prompts without selection (should be rejected)
- Test instance deletion (should remove all associated prompts)
- Test mask clipping (masks should not extend beyond bounding boxes)
- Test mask overlap prevention (new masks should not overlap existing ones)

## Bug Fixes (Latest)

### Fixed: Point prompt marker persists when clicking on existing mask

- **Issue**: When clicking on an existing mask with a point prompt, the prompt marker was still being created even though no new instance was created
- **Fix**: Only add prompt marker to `prompted_points` when creating a new instance OR when explicitly refining (not auto-detected). Auto-detected refinements don't show visual markers since they're just refining the existing instance.

### Fixed: Array dimension mismatch error

- **Issue**: Error "all the input arrays must have same number of dimensions" when clicking on existing masks
- **Fix**:
  - Ensure all masks have consistent shapes before concatenation
  - Convert masks to numpy arrays and ensure they're all 2D
  - Use `np.stack()` instead of `np.concatenate()` for masks to ensure proper shape handling
  - Handle shape mismatches by resizing masks to match target shape
  - Fixed dimension mismatch in processor by normalizing mask shapes (handle 2D, 3D, and 4D shapes)

### Fixed: Multiple instances created from single click

- **Issue**: When clicking to create a new instance, multiple instances were being created instead of just one
- **Fix**:
  - Added check in both point and box prompt handlers to ensure only one instance is created
  - If processor returns multiple instances, keep only the best one (highest confidence score)
  - Applied fix to both point and box prompt creation paths
