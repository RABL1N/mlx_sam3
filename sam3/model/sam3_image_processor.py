import time
from functools import partial

from typing import Dict, List
import PIL
from PIL import Image
import numpy as np
import mlx.core as mx

from sam3.model import box_ops
from sam3.model.data_misc import FindStage, interpolate

# TODO: remove this, using for testing
import torch
from torchvision.transforms import v2


def transform(image_path_or_pil, resolution):
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")
    
    img = img.resize((resolution, resolution), resample=Image.Resampling.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0 # [H, W, C]

    img_np = (img_np - 0.5) / 0.5

    return mx.array(img_np).transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]

class Sam3Processor:
    def __init__(self, model, resolution=1008, confidence_threshold=0.5):
        self.model = model
        self.resolution = resolution
        self.confidence_threshold = confidence_threshold
        self.transform = partial(transform, resolution=self.resolution)


        self.find_stage = FindStage(
            img_ids=mx.array([0], dtype=mx.int64),
            text_ids=mx.array([0], dtype=mx.int64),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

   
    def set_image(self, image, state=None):
        if state is None:
            state = {}
        
        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        # elif isinstance(image, (mx.array, np.ndarray)):
        #     height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image")
        
        image = self.transform(image)[None]

        state["original_height"] = height
        state["original_width"] = width
        import time
        start = time.perf_counter()
        state["backbone_out"] = self.model.backbone.call_image(image)
        mx.eval(state)
        second = time.perf_counter()
        print(f"Backbone pass took {second - start:.2f} Seconds")
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    def set_image_batch(self, iamges: List[np.ndarray], state=None):
        pass

    def set_text_prompt(self, prompt: str, state: Dict):
        """Sets a text prompt for grounding-based segmentation.
        
        Note: This will find ALL instances matching the text prompt in the image.
        If you want to segment a specific instance, use point or box prompts instead.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")
        
        text_outputs = self.model.backbone.call_text([prompt])
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        # Reset geometric prompts when setting a new text prompt
        # This ensures text prompts work independently of previous geometric prompts
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()
        else:
            # Reset to empty geometric prompt when using text prompt
            state["geometric_prompt"] = self.model._get_dummy_prompt()
        return self._call_grounding(state)

    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        
        Note: This will clear any existing text prompt and reset geometric prompts to ensure
        only the specified box is used for segmentation (not all instances matching a text prompt).
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_geometric_prompt")

        # Clear any existing text prompt to avoid finding all instances matching the text
        # Use "visual" instead so the model relies only on the geometric prompt
        dummy_text_outputs = self.model.backbone.call_text(["visual"])
        state["backbone_out"].update(dummy_text_outputs)

        # Reset geometric prompts so we only use the latest box (not accumulated boxes)
        state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = mx.array(box, dtype=mx.float32).reshape(1, 1, 4)
        labels = mx.array([label], dtype=mx.bool_).reshape(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._call_grounding(state)

    def add_point_prompt(self, point: List, label: bool, state: Dict):
        """Adds a point prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The point is assumed to be in [x, y] format and normalized in [0, 1] range.
        The label is True for a positive point, False for a negative point.
        
        Note: This will clear any existing text prompt and reset geometric prompts to ensure
        only the specified point is used for segmentation (not all instances matching a text prompt).
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_point_prompt")

        # Clear any existing text prompt to avoid finding all instances matching the text
        # Use "visual" instead so the model relies only on the geometric prompt
        dummy_text_outputs = self.model.backbone.call_text(["visual"])
        state["backbone_out"].update(dummy_text_outputs)

        # Reset geometric prompts so we only use the latest point (not accumulated points)
        state["geometric_prompt"] = self.model._get_dummy_prompt()

        # Points need to be in [x, y] format, normalized [0, 1]
        # Shape: [num_points, batch_size, 2]
        points = mx.array(point, dtype=mx.float32).reshape(1, 1, 2)
        # Labels: 1 for positive, 0 for negative
        # Use int32 instead of int64 because MLX scatter doesn't support int64
        labels = mx.array([1 if label else 0], dtype=mx.int32).reshape(1, 1)
        
        state["geometric_prompt"].append_points(points, labels)
        return self._call_grounding(state)

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    def set_confidence_threshold(self, threshold: float, state=None):
        pass

    def _call_grounding(self, state: Dict):
        outputs = self.model.call_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = mx.sigmoid(out_logits)
        presence_score = mx.sigmoid(outputs["presence_logit_dec"])[:,None]
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        mask_np = np.array(keep[0])
        indices = mx.array(mask_np.nonzero()[0])
        out_probs = out_probs[0][indices]
        # out_probs = out_probs[keep]
        out_masks = out_masks[0][indices]
        out_bbox = out_bbox[0][indices]
        seg_mask = outputs['semantic_seg']

        # convert box to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = mx.array([img_w, img_h, img_w, img_h])
        boxes = boxes * scale_fct[None, :]

        interpolator = partial(interpolate,
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        )
        out_masks = interpolator(out_masks[:, None])
        out_masks = mx.sigmoid(out_masks)

        seg_mask = interpolator(seg_mask)

        # Filter to single instance if geometric prompts are present
        geometric_prompt = state.get("geometric_prompt")
        if geometric_prompt is not None:
            try:
                # Check if we have points or boxes in the geometric prompt
                # Convert shape to Python int to avoid MLX array comparison issues
                has_points = False
                if geometric_prompt.point_embeddings is not None:
                    point_shape = geometric_prompt.point_embeddings.shape
                    if isinstance(point_shape, tuple):
                        has_points = len(point_shape) > 0 and int(point_shape[0]) > 0
                    else:
                        # If shape is an array, convert it
                        point_shape_np = np.array(point_shape)
                        has_points = len(point_shape_np) > 0 and int(point_shape_np[0]) > 0
                
                has_boxes = False
                if geometric_prompt.box_embeddings is not None:
                    box_shape = geometric_prompt.box_embeddings.shape
                    if isinstance(box_shape, tuple):
                        has_boxes = len(box_shape) > 0 and int(box_shape[0]) > 0
                    else:
                        # If shape is an array, convert it
                        box_shape_np = np.array(box_shape)
                        has_boxes = len(box_shape_np) > 0 and int(box_shape_np[0]) > 0
                
                if has_points or has_boxes:
                    # Get the prompt location in pixel coordinates
                    if has_points:
                        # Use the first positive point
                        point_emb = geometric_prompt.point_embeddings[0, 0]  # [2]
                        point_labels = geometric_prompt.point_labels[0, 0] if geometric_prompt.point_labels is not None else mx.array(1)
                        # Convert MLX arrays to numpy arrays first, then extract scalars
                        point_emb_np = np.array(point_emb)
                        point_label_val = int(np.array(point_labels))
                        if point_label_val > 0:  # Only use positive points
                            prompt_x = float(point_emb_np[0]) * img_w
                            prompt_y = float(point_emb_np[1]) * img_h
                            
                            # Find the instance whose mask contains the point or box center is closest
                            best_idx = None
                            best_score = -1
                            found_in_mask = False
                            boxes_np = np.array(boxes)
                            
                            # First pass: check if point is inside any mask
                            for i in range(len(boxes_np)):
                                x0, y0, x1, y1 = boxes_np[i]
                                
                                # Check if point is inside the box
                                if x0 <= prompt_x <= x1 and y0 <= prompt_y <= y1:
                                    # Point is inside box - check mask
                                    mask_np = np.array(out_masks[i] > 0.5)
                                    # Handle different mask shapes (could be 2D or 3D)
                                    if len(mask_np.shape) == 3:
                                        mask_np = mask_np[0]  # Take first channel if 3D
                                    if len(mask_np.shape) == 2 and mask_np.shape[0] > 0 and mask_np.shape[1] > 0:
                                        y_idx = int(np.clip(prompt_y / img_h * mask_np.shape[0], 0, mask_np.shape[0] - 1))
                                        x_idx = int(np.clip(prompt_x / img_w * mask_np.shape[1], 0, mask_np.shape[1] - 1))
                                        # Get scalar value from array
                                        mask_value = bool(mask_np[y_idx, x_idx])
                                        if mask_value:
                                            # Point is in mask - this is the best match
                                            best_idx = i
                                            best_score = float(out_probs[i])
                                            found_in_mask = True
                                            break
                            
                            # Second pass: if not found in mask, find closest by distance
                            if not found_in_mask:
                                for i in range(len(boxes_np)):
                                    x0, y0, x1, y1 = boxes_np[i]
                                    box_center_x = (x0 + x1) / 2
                                    box_center_y = (y0 + y1) / 2
                                    
                                    # Calculate distance from point to box center
                                    dist = np.sqrt((prompt_x - box_center_x)**2 + (prompt_y - box_center_y)**2)
                                    # Score combines confidence and proximity (closer = better)
                                    score = float(out_probs[i]) / (1.0 + dist / max(img_w, img_h))
                                    if score > best_score:
                                        best_score = score
                                        best_idx = i
                            
                            # Always filter to single instance when we have a point prompt
                            if len(boxes_np) > 0:
                                if best_idx is None:
                                    # Fallback: use first instance if somehow best_idx is still None
                                    best_idx = 0
                                
                                if best_idx < len(boxes_np):
                                    # Keep only the best matching instance
                                    new_probs = out_probs[best_idx:best_idx+1]
                                    new_masks = out_masks[best_idx:best_idx+1]
                                    new_boxes = boxes[best_idx:best_idx+1]
                                    
                                    # Append to existing masks/boxes/scores if they exist
                                    if "masks" in state and state["masks"] is not None and len(state["masks"]) > 0:
                                        # Get existing masks - prefer mask_logits (float) if available, otherwise convert boolean masks to float
                                        if "mask_logits" in state and state["mask_logits"] is not None:
                                            existing_masks = state["mask_logits"]
                                        else:
                                            # Convert boolean masks to float for concatenation
                                            existing_masks = state["masks"].astype(mx.float32)
                                        existing_boxes = state.get("boxes", mx.zeros((0, 4)))
                                        existing_scores = state.get("scores", mx.zeros((0,)))
                                        
                                        # Ensure shapes are compatible for concatenation
                                        if len(existing_masks.shape) == 2:
                                            existing_masks = existing_masks[None]  # Add batch dimension
                                        if len(existing_boxes.shape) == 1:
                                            existing_boxes = existing_boxes[None]
                                        if len(existing_scores.shape) == 0:
                                            existing_scores = existing_scores[None]
                                        
                                        out_masks = mx.concatenate([existing_masks, new_masks], axis=0)
                                        boxes = mx.concatenate([existing_boxes, new_boxes], axis=0)
                                        out_probs = mx.concatenate([existing_scores, new_probs], axis=0)
                                    else:
                                        # First instance, just use the new one
                                        out_probs = new_probs
                                        out_masks = new_masks
                                        boxes = new_boxes
                            else:
                                # No match found, keep existing masks if any
                                if "masks" in state and state["masks"] is not None and len(state["masks"]) > 0:
                                    # Use mask_logits if available, otherwise convert boolean to float
                                    if "mask_logits" in state and state["mask_logits"] is not None:
                                        out_masks = state["mask_logits"]
                                    else:
                                        out_masks = state["masks"].astype(mx.float32)
                                    boxes = state.get("boxes", mx.zeros((0, 4)))
                                    out_probs = state.get("scores", mx.zeros((0,)))
                        else:
                            # No positive point found, keep existing masks if any
                            if "masks" in state and state["masks"] is not None and len(state["masks"]) > 0:
                                # Use mask_logits if available, otherwise convert boolean to float
                                if "mask_logits" in state and state["mask_logits"] is not None:
                                    out_masks = state["mask_logits"]
                                else:
                                    out_masks = state["masks"].astype(mx.float32)
                                boxes = state.get("boxes", mx.zeros((0, 4)))
                                out_probs = state.get("scores", mx.zeros((0,)))
                    
                    elif has_boxes:
                        # Use the first positive box
                        box_emb = geometric_prompt.box_embeddings[0, 0]  # [4] cxcywh
                        box_labels = geometric_prompt.box_labels[0, 0] if geometric_prompt.box_labels is not None else mx.array(True)
                        # Convert MLX arrays to numpy arrays first, then extract scalars
                        box_emb_np = np.array(box_emb)
                        box_label_val = bool(np.array(box_labels))
                        if box_label_val:  # Only use positive boxes
                            cx, cy, w, h = float(box_emb_np[0]), float(box_emb_np[1]), float(box_emb_np[2]), float(box_emb_np[3])
                            prompt_x0 = (cx - w/2) * img_w
                            prompt_y0 = (cy - h/2) * img_h
                            prompt_x1 = (cx + w/2) * img_w
                            prompt_y1 = (cy + h/2) * img_h
                            
                            # Find the instance with highest IoU with the prompt box
                            best_idx = None
                            best_iou = -1
                            boxes_np = np.array(boxes)
                            
                            for i in range(len(boxes_np)):
                                pred_box = boxes_np[i]
                                # Calculate IoU
                                x0_int = max(prompt_x0, pred_box[0])
                                y0_int = max(prompt_y0, pred_box[1])
                                x1_int = min(prompt_x1, pred_box[2])
                                y1_int = min(prompt_y1, pred_box[3])
                                
                                if x1_int > x0_int and y1_int > y0_int:
                                    intersection = (x1_int - x0_int) * (y1_int - y0_int)
                                    prompt_area = (prompt_x1 - prompt_x0) * (prompt_y1 - prompt_y0)
                                    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                                    union = prompt_area + pred_area - intersection
                                    iou = intersection / union if union > 0 else 0
                                    
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_idx = i
                            
                            if best_idx is not None and len(boxes_np) > 0 and best_idx < len(boxes_np):
                                # Keep only the best matching instance
                                new_probs = out_probs[best_idx:best_idx+1]
                                new_masks = out_masks[best_idx:best_idx+1]
                                new_boxes = boxes[best_idx:best_idx+1]
                                
                                # Append to existing masks/boxes/scores if they exist
                                if "masks" in state and state["masks"] is not None and len(state["masks"]) > 0:
                                    # Concatenate with existing masks
                                    existing_masks = state["masks"]
                                    existing_boxes = state.get("boxes", mx.zeros((0, 4)))
                                    existing_scores = state.get("scores", mx.zeros((0,)))
                                    
                                    # Ensure shapes are compatible for concatenation
                                    if len(existing_masks.shape) == 2:
                                        existing_masks = existing_masks[None]  # Add batch dimension
                                    if len(existing_boxes.shape) == 1:
                                        existing_boxes = existing_boxes[None]
                                    if len(existing_scores.shape) == 0:
                                        existing_scores = existing_scores[None]
                                    
                                    out_masks = mx.concatenate([existing_masks, new_masks], axis=0)
                                    boxes = mx.concatenate([existing_boxes, new_boxes], axis=0)
                                    out_probs = mx.concatenate([existing_scores, new_probs], axis=0)
                                else:
                                    # First instance, just use the new one
                                    out_probs = new_probs
                                    out_masks = new_masks
                                    boxes = new_boxes
                            else:
                                # No match found, keep existing masks if any
                                if "masks" in state and state["masks"] is not None and len(state["masks"]) > 0:
                                    out_masks = state["masks"]
                                    boxes = state.get("boxes", mx.zeros((0, 4)))
                                    out_probs = state.get("scores", mx.zeros((0,)))
                        else:
                            # No positive point found, keep existing masks if any
                            if "masks" in state and state["masks"] is not None and len(state["masks"]) > 0:
                                out_masks = state["masks"]
                                boxes = state.get("boxes", mx.zeros((0, 4)))
                                out_probs = state.get("scores", mx.zeros((0,)))
                    
            except Exception as e:
                # If filtering fails, just use all results (fallback to original behavior)
                import traceback
                print(f"Warning: Failed to filter to single instance: {e}")
                print(traceback.format_exc())
                pass

        state["semantic_seg"] = seg_mask
        state["mask_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state