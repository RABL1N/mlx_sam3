"""
FastAPI backend for SAM3 segmentation model.
Provides endpoints for image upload, text prompts, box prompts, and segmentation results.
"""

import io
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
import mlx.core as mx

# Add parent directory to path to import sam3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Global model and processor
model = None
processor = None

# Session storage for processing states
sessions: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, processor
    
    sam3_root = os.path.dirname(sam3.__file__)
    # bpe_path = os.path.join(sam3_root, "..", "assets", "bpe_simple_vocab_16e6.txt.gz")
    # checkpoint_path = os.path.join(sam3_root, "..", "sam3-mod-weights", "model.safetensors")
    
    # print(f"Loading SAM3 model from {checkpoint_path}...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("SAM3 model loaded successfully!")
    
    yield
    
    # Cleanup
    sessions.clear()


app = FastAPI(
    title="SAM3 Segmentation API",
    description="API for interactive image segmentation using SAM3 model",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextPromptRequest(BaseModel):
    session_id: str
    prompt: str


class BoxPromptRequest(BaseModel):
    session_id: str
    box: list[float]  # [center_x, center_y, width, height] normalized
    label: bool  # True for positive, False for negative


class PointPromptRequest(BaseModel):
    session_id: str
    point: list[float]  # [x, y] normalized in [0, 1]
    label: bool  # True for positive, False for negative


class ConfidenceRequest(BaseModel):
    session_id: str
    threshold: float


class SessionRequest(BaseModel):
    session_id: str

class RemoveInstanceRequest(BaseModel):
    session_id: str
    instance_index: int


class UpdateCategoryRequest(BaseModel):
    session_id: str
    instance_index: int
    category_id: int | None  # None to unassign category



def mask_to_rle(mask: np.ndarray) -> dict:
    """
    Encode a binary mask to RLE (Run-Length Encoding) format.
    
    Args:
        mask: 2D binary numpy array (H, W) with values 0 or 1
        
    Returns:
        dict with 'counts' (list of run lengths) and 'size' [H, W]
    """
    # Flatten the mask in row-major (C) order
    flat = mask.flatten()
    
    # Find where values change
    diff = np.diff(flat)
    change_indices = np.where(diff != 0)[0] + 1
    
    # Build run lengths
    run_starts = np.concatenate([[0], change_indices])
    run_ends = np.concatenate([change_indices, [len(flat)]])
    run_lengths = (run_ends - run_starts).tolist()
    
    # If mask starts with 1, prepend a 0-length run for background
    if flat[0] == 1:
        run_lengths = [0] + run_lengths
    
    return {
        "counts": run_lengths,
        "size": list(mask.shape)  # [H, W]
    }


def serialize_state(state: dict) -> dict:
    """Convert state arrays to JSON-serializable format."""
    result = {
        "original_width": state.get("original_width"),
        "original_height": state.get("original_height"),
    }
    
    if "masks" in state:
        masks = state["masks"]
        boxes = state["boxes"]
        scores = state["scores"]
        
        masks_list = []
        boxes_list = []
        scores_list = []
        
        for i in range(len(scores)):
            mask_np = to_numpy_array(masks[i])
            # Handle boxes - they might be MLX arrays or lists
            # If boxes is an MLX array, convert to numpy first, then index
            if hasattr(boxes, 'shape') and hasattr(boxes, '__array__'):
                # It's an MLX or numpy array
                boxes_np = np.array(boxes)
                box_np = boxes_np[i]
            else:
                # It's a list
                box_np = to_numpy_array(boxes[i])
            score_arr = to_numpy_array(scores[i])
            # Safely convert score to float (handle arrays)
            try:
                if score_arr.size == 0:
                    score_np = 0.0
                elif score_arr.size == 1:
                    # Single element array - use item() to get scalar
                    score_np = float(score_arr.item())
                else:
                    # Multiple elements - take first one
                    score_np = float(score_arr.flat[0])
            except (AttributeError, ValueError, TypeError):
                # Fallback: try direct conversion
                try:
                    score_np = float(score_arr)
                except (ValueError, TypeError):
                    score_np = 0.0
            
            # Convert mask to binary and get the 2D mask (handle [1, H, W] shape)
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            if mask_binary.ndim == 3:
                mask_binary = mask_binary[0]  # Take first channel
            
            # Encode as RLE
            rle = mask_to_rle(mask_binary)
            masks_list.append(rle)
            boxes_list.append(box_np.tolist())
            scores_list.append(score_np)
        
        result["masks"] = masks_list
        result["boxes"] = boxes_list
        # Get category_ids if they exist, otherwise default to None for all
        category_ids = state.get("category_ids", [None] * len(scores))
        if len(category_ids) < len(scores):
            # Pad with None if category_ids is shorter
            category_ids.extend([None] * (len(scores) - len(category_ids)))
        
        category_ids_list = []
        for i in range(len(scores)):
            category_id = category_ids[i] if i < len(category_ids) else None
            category_ids_list.append(category_id)
        
        result["scores"] = scores_list
        result["category_ids"] = category_ids_list
    
    if "prompted_boxes" in state:
        result["prompted_boxes"] = state["prompted_boxes"]
    
    if "prompted_points" in state:
        result["prompted_points"] = state["prompted_points"]
    
    return result


@app.get("/")
async def root():
    return {"message": "SAM3 Segmentation API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image and initialize a session."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Create session
        session_id = str(uuid.uuid4())
        
        # Process image through model (timed)
        start_time = time.perf_counter()
        state = processor.set_image(image)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Store session with image info
        sessions[session_id] = {
            "state": state,
            "image_size": image.size,
            "image": image,  # Store PIL image for visualizations
            "image_filename": file.filename or "image.jpg",
        }
        
        return {
            "session_id": session_id,
            "width": image.size[0],
            "height": image.size[1],
            "message": "Image uploaded and processed successfully",
            "processing_time_ms": round(processing_time_ms, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/segment/text")
async def segment_with_text(request: TextPromptRequest):
    """Segment image using text prompt."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        start_time = time.perf_counter()
        state = processor.set_text_prompt(request.prompt, session["state"])
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        session["state"] = state
        start = time.perf_counter()
        results = serialize_state(state)
        end = time.perf_counter()
        print(f"Serialization took {end - start:.4f} seconds")
        
        return {
            "session_id": request.session_id,
            "prompt": request.prompt,
            "results": results,
            "processing_time_ms": round(processing_time_ms, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during segmentation: {str(e)}")


@app.post("/segment/box")
async def add_box_prompt(request: BoxPromptRequest):
    """Add a box prompt (positive or negative) and re-segment."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        state = session["state"]
        
        # Convert from normalized cxcywh to pixel xyxy for IoU check
        img_w = state["original_width"]
        img_h = state["original_height"]
        cx, cy, w, h = request.box
        x_min = (cx - w / 2) * img_w
        y_min = (cy - h / 2) * img_h
        x_max = (cx + w / 2) * img_w
        y_max = (cy + h / 2) * img_h
        new_box_xyxy = np.array([x_min, y_min, x_max, y_max])
        
        # Check IoU with existing boxes if we have instances
        if "boxes" in state and state["boxes"] is not None:
            boxes_array = np.array(state["boxes"])
            if boxes_array.size > 0:
                for existing_box in boxes_array:
                    existing_box_np = to_numpy_array(existing_box)
                    iou = calculate_box_iou(new_box_xyxy, existing_box_np)
                    # Ensure iou is a scalar float
                    iou_val = float(iou) if not isinstance(iou, (int, float)) else float(iou)
                    if iou_val > 0.3:
                        # Silently reject - don't create instance and don't add prompted box
                        return {
                            "session_id": request.session_id,
                            "box_type": "positive" if request.label else "negative",
                            "results": serialize_state(state),
                            "processing_time_ms": 0.0
                        }
        
        # Store prompted box for display (only if we pass the IoU check)
        if "prompted_boxes" not in state:
            state["prompted_boxes"] = []
        
        state["prompted_boxes"].append({
            "box": [x_min, y_min, x_max, y_max],
            "label": request.label
        })
        
        start_time = time.perf_counter()
        state = processor.add_geometric_prompt(request.box, request.label, state)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Remove mask overlap (earlier masks take priority)
        if "masks" in state and state["masks"] is not None and len(state["masks"]) > 0:
            state["masks"] = remove_mask_overlap(state["masks"], state.get("boxes", []))
        
        # Recalculate tight bounding boxes after segmentation
        if "masks" in state and state["masks"] is not None and len(state["masks"]) > 0:
            if "boxes" not in state:
                state["boxes"] = []
            if "category_ids" not in state:
                state["category_ids"] = []
            
            # Get number of masks (handle both MLX arrays and lists)
            num_masks = state["masks"].shape[0] if hasattr(state["masks"], 'shape') else len(state["masks"])
            
            # Ensure category_ids list is long enough
            while len(state["category_ids"]) < num_masks:
                state["category_ids"].append(None)
            
            # Recalculate tight bboxes for all masks
            # Keep boxes as MLX arrays for the processor
            new_boxes = []
            # Convert masks to numpy for processing
            masks_np = np.array(state["masks"])
            # Handle 4D format [N, 1, H, W] - squeeze channel dimension if present
            if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                masks_np = masks_np[:, 0, :, :]  # Remove channel dimension: [N, H, W]
            for i in range(num_masks):
                mask = masks_np[i]
                tight_bbox = calculate_tight_bbox_from_mask(mask)
                # Convert numpy array to list for MLX
                new_boxes.append(tight_bbox.tolist())
            
            # Convert to MLX array if we have boxes, otherwise use empty array
            if new_boxes:
                state["boxes"] = mx.array(new_boxes)
            else:
                state["boxes"] = mx.zeros((0, 4))
        
        session["state"] = state
        
        return {
            "session_id": request.session_id,
            "box_type": "positive" if request.label else "negative",
            "results": serialize_state(state),
            "processing_time_ms": round(processing_time_ms, 2)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding box prompt: {str(e)}")


@app.post("/reset")
async def reset_prompts(request: SessionRequest):
    """Reset all prompts for a session."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        state = session["state"]
        
        # Preserve masks, boxes, scores, and category_ids before resetting prompts
        preserved_masks = state.get("masks")
        preserved_boxes = state.get("boxes")
        preserved_scores = state.get("scores")
        preserved_category_ids = state.get("category_ids")
        
        start_time = time.perf_counter()
        processor.reset_all_prompts(state)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Restore masks, boxes, scores, and category_ids if they existed
        if preserved_masks is not None:
            state["masks"] = preserved_masks
        if preserved_boxes is not None:
            state["boxes"] = preserved_boxes
        if preserved_scores is not None:
            state["scores"] = preserved_scores
        if preserved_category_ids is not None:
            state["category_ids"] = preserved_category_ids
        
        # Clear all prompts
        if "prompted_boxes" in state:
            del state["prompted_boxes"]
        if "prompted_points" in state:
            state["prompted_points"] = []
        
        return {
            "session_id": request.session_id,
            "message": "All prompts reset",
            "results": serialize_state(state),
            "processing_time_ms": round(processing_time_ms, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting prompts: {str(e)}")


@app.post("/confidence")
async def set_confidence(request: ConfidenceRequest):
    """Update confidence threshold (note: requires re-running inference)."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Update processor threshold
    processor.confidence_threshold = request.threshold
    
    return {
        "session_id": request.session_id,
        "threshold": request.threshold,
        "message": "Confidence threshold updated. Re-run segmentation to apply."
    }


@app.post("/segment/point")
async def add_point_prompt(request: PointPromptRequest):
    """Add a point prompt (positive or negative) and re-segment."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        state = session["state"]
        
        # Check if point is on an existing mask BEFORE processing
        # If it is, silently reject (existing mask takes priority)
        if "masks" in state and state["masks"] is not None:
            masks_array = np.array(state["masks"])
            if masks_array.size > 0:
                # Handle 4D format [N, 1, H, W]
                if masks_array.ndim == 4 and masks_array.shape[1] == 1:
                    masks_array = masks_array[:, 0, :, :]  # [N, H, W]
                
                img_w = state["original_width"]
                img_h = state["original_height"]
                
                # Check if point is on any existing mask
                for i in range(masks_array.shape[0]):
                    mask = masks_array[i]
                    if check_point_on_mask(request.point, mask, img_w, img_h):
                        # Point is on existing mask - silently reject
                        return {
                            "session_id": request.session_id,
                            "point_type": "positive" if request.label else "negative",
                            "results": serialize_state(state),
                            "processing_time_ms": 0.0
                        }
        
        start_time = time.perf_counter()
        state = processor.add_point_prompt(request.point, request.label, state)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Store prompted point for display (only if we get here, meaning it wasn't rejected)
        if "prompted_points" not in state:
            state["prompted_points"] = []
        state["prompted_points"].append({
            "point": request.point,
            "label": request.label
        })
        
        # Remove mask overlap (earlier masks take priority)
        if "masks" in state and state["masks"] is not None and len(state["masks"]) > 0:
            state["masks"] = remove_mask_overlap(state["masks"], state.get("boxes", []))
        
        # Recalculate tight bounding boxes after segmentation
        if "masks" in state and state["masks"] is not None and len(state["masks"]) > 0:
            if "boxes" not in state:
                state["boxes"] = []
            if "category_ids" not in state:
                state["category_ids"] = []
            
            # Get number of masks (handle both MLX arrays and lists)
            num_masks = state["masks"].shape[0] if hasattr(state["masks"], 'shape') else len(state["masks"])
            
            # Ensure category_ids list is long enough
            while len(state["category_ids"]) < num_masks:
                state["category_ids"].append(None)
            
            # Recalculate tight bboxes for all masks
            # Keep boxes as MLX arrays for the processor
            new_boxes = []
            # Convert masks to numpy for processing
            masks_np = np.array(state["masks"])
            # Handle 4D format [N, 1, H, W] - squeeze channel dimension if present
            if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                masks_np = masks_np[:, 0, :, :]  # Remove channel dimension: [N, H, W]
            for i in range(num_masks):
                mask = masks_np[i]
                tight_bbox = calculate_tight_bbox_from_mask(mask)
                # Convert numpy array to list for MLX
                new_boxes.append(tight_bbox.tolist())
            
            # Convert to MLX array if we have boxes, otherwise use empty array
            if new_boxes:
                state["boxes"] = mx.array(new_boxes)
            else:
                state["boxes"] = mx.zeros((0, 4))
            
            # Check IoU with existing boxes (excluding the newly created one)
            if state["boxes"].shape[0] > 1:
                # Convert to numpy for IoU calculation
                boxes_np = np.array(state["boxes"])
                new_box = boxes_np[-1]
                for i in range(boxes_np.shape[0] - 1):
                    existing_box = boxes_np[i]
                    iou = calculate_box_iou(new_box, existing_box)
                    # Ensure iou is a scalar float
                    iou_val = float(iou) if not isinstance(iou, (int, float)) else float(iou)
                    if iou_val > 0.3:
                        # Remove the new instance if it overlaps too much
                        state["masks"] = state["masks"][:-1]
                        # Remove last box from MLX array
                        state["boxes"] = state["boxes"][:-1]
                        if state.get("category_ids") and len(state["category_ids"]) > 0:
                            state["category_ids"] = state["category_ids"][:-1]
                        if "scores" in state and state["scores"] is not None:
                            # Check if scores has length > 0 (handle both MLX arrays and lists)
                            scores_len = state["scores"].shape[0] if hasattr(state["scores"], 'shape') else len(state["scores"])
                            if scores_len > 0:
                                state["scores"] = state["scores"][:-1]
                        # Remove the point from prompted_points since we're rejecting it
                        if "prompted_points" in state and state["prompted_points"]:
                            state["prompted_points"] = state["prompted_points"][:-1]
                        session["state"] = state
                        return {
                            "session_id": request.session_id,
                            "point_type": "positive" if request.label else "negative",
                            "results": serialize_state(state),
                            "processing_time_ms": processing_time_ms
                        }
        
        session["state"] = state
        
        return {
            "session_id": request.session_id,
            "point_type": "positive" if request.label else "negative",
            "results": serialize_state(state),
            "processing_time_ms": round(processing_time_ms, 2)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in add_point_prompt: {str(e)}")
        print(f"Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Error adding point prompt: {str(e)}")


@app.post("/save-annotations")
async def save_annotations(request: SessionRequest):
    """Save current session annotations in COCO format to server filesystem."""
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        state = session["state"]
        results = serialize_state(state)
        
        if not results.get("masks") or len(results["masks"]) == 0:
            raise HTTPException(status_code=400, detail="No masks to save")
        
        # Determine output directory (relative to mlx_sam3 root)
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(backend_dir))  # Go up to mlx_sam3
        output_dir = os.path.join(project_root, "annotations")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp-based directory name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(output_dir, f"session_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Get image filename from session if available
        image_filename = session.get("image_filename", "image.jpg")
        image_id = int(os.path.splitext(os.path.basename(image_filename))[0]) if os.path.splitext(os.path.basename(image_filename))[0].isdigit() else 1
        
        # Save individual mask images
        saved_files = []
        for idx, (mask_rle, box, score) in enumerate(zip(
            results["masks"],
            results.get("boxes", []),
            results.get("scores", [])
        )):
            # Decode RLE to binary mask
            mask_binary = rle_to_mask(mask_rle)
            
            # Save mask PNG
            mask_img = Image.fromarray((mask_binary * 255).astype(np.uint8), mode="L")
            mask_filename = f"{image_id}_instance_{idx:03d}_mask.png"
            mask_path = os.path.join(session_dir, mask_filename)
            mask_img.save(mask_path)
            saved_files.append(mask_filename)
            
            # Create visualization with overlay (if we have the original image)
            if "image" in session:
                vis_img = session["image"].copy().convert("RGB")
                mask_h, mask_w = mask_binary.shape
                img_w, img_h = vis_img.size
                
                if (mask_h, mask_w) != (img_h, img_w):
                    mask_pil = Image.fromarray((mask_binary * 255).astype(np.uint8), mode="L")
                    mask_pil = mask_pil.resize((img_w, img_h), Image.BILINEAR)
                    mask_binary = np.array(mask_pil) / 255.0
                
                # Create colored overlay
                overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
                overlay[..., 0] = 255  # R
                overlay[..., 1] = 0    # G
                overlay[..., 2] = 0    # B
                overlay[..., 3] = (mask_binary * 128).astype(np.uint8)
                
                vis_img_rgba = vis_img.convert("RGBA")
                overlay_img = Image.fromarray(overlay, mode="RGBA")
                vis_img = Image.alpha_composite(vis_img_rgba, overlay_img).convert("RGB")
                
                # Draw bounding box
                from PIL import ImageDraw
                draw = ImageDraw.Draw(vis_img)
                x0, y0, x1, y1 = box
                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                
                vis_filename = f"{image_id}_instance_{idx:03d}_vis.png"
                vis_path = os.path.join(session_dir, vis_filename)
                vis_img.save(vis_path)
                saved_files.append(vis_filename)
        
        # Define fungus categories (same as in annotate.py)
        categories = [
            {"id": 1, "name": "aspergillus", "supercategory": "fungus"},
            {"id": 2, "name": "penicillium", "supercategory": "fungus"},
            {"id": 3, "name": "rhizopus", "supercategory": "fungus"},
            {"id": 4, "name": "mucor", "supercategory": "fungus"},
            {"id": 5, "name": "other_fungus", "supercategory": "fungus"},
        ]
        
        # Create COCO format JSON
        coco_data = {
            "info": {
                "description": "SAM3 Instance Segmentation Annotations",
                "version": "1.0",
                "year": time.localtime().tm_year,
            },
            "licenses": [],
            "images": [{
                "id": image_id,
                "width": results["original_width"],
                "height": results["original_height"],
                "file_name": image_filename,
            }],
            "annotations": [
                {
                    "id": idx + 1,
                    "image_id": image_id,
                    "category_id": results.get("category_ids", [None] * len(results["masks"]))[idx] if idx < len(results.get("category_ids", [])) else None,
                    "segmentation": mask_rle,
                    "bbox": box,
                    "area": int(rle_to_mask(mask_rle).sum()),
                    "iscrowd": 0,
                    "score": float(score),
                    "instance_id": idx,
                }
                for idx, (mask_rle, box, score) in enumerate(zip(
                    results["masks"],
                    results.get("boxes", []),
                    results.get("scores", [])
                ))
            ],
            "categories": categories,
        }
        
        # Save JSON file
        json_filename = f"{image_id}_annotations.json"
        json_path = os.path.join(session_dir, json_filename)
        import json
        with open(json_path, "w") as f:
            json.dump(coco_data, f, indent=2)
        saved_files.append(json_filename)
        
        # Return relative path from project root
        relative_path = os.path.relpath(session_dir, project_root)
        
        return {
            "session_id": request.session_id,
            "message": f"Annotations saved successfully",
            "output_directory": relative_path,
            "files_saved": saved_files,
            "annotations": results,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving annotations: {str(e)}")


def to_numpy_array(arr):
    """Safely convert MLX arrays or other types to numpy arrays."""
    try:
        # Check if it's an MLX array (has __array__ or specific MLX attributes)
        if hasattr(arr, '__array__'):
            # Use __array__ if available (works for both numpy and MLX)
            return np.asarray(arr)
        elif hasattr(arr, 'tolist'):
            # MLX arrays have tolist() method
            result = np.array(arr.tolist())
            return result
        else:
            # Already numpy or can be converted directly
            return np.asarray(arr)
    except (AttributeError, TypeError, ValueError) as e:
        # Fallback: try direct conversion
        try:
            return np.asarray(arr)
        except:
            # Last resort: try tolist if available
            if hasattr(arr, 'tolist'):
                return np.array(arr.tolist())
            raise ValueError(f"Could not convert {type(arr)} to numpy array: {e}")


def rle_to_mask(rle: dict) -> np.ndarray:
    """Convert RLE dict to binary mask numpy array."""
    height, width = rle["size"]
    mask = np.zeros(height * width, dtype=np.uint8)
    
    counts = rle["counts"]
    if isinstance(counts, str):
        # Parse space-separated string
        counts = [int(x) for x in counts.split() if x]
    
    pixel_idx = 0
    is_foreground = False
    
    for count in counts:
        if is_foreground:
            end_idx = min(pixel_idx + count, len(mask))
            mask[pixel_idx:end_idx] = 1
            pixel_idx = end_idx
        else:
            pixel_idx += count
        is_foreground = not is_foreground
    
    return mask.reshape((height, width))


def check_point_on_mask(point: list[float], mask: np.ndarray, img_w: int, img_h: int) -> bool:
    """
    Check if a normalized point [x, y] in [0, 1] is on a mask.
    
    Args:
        point: [x, y] normalized coordinates in [0, 1]
        mask: 2D or 3D mask array
        img_w: Original image width
        img_h: Original image height
        
    Returns:
        True if point is on the mask, False otherwise
    """
    # Convert normalized point to pixel coordinates
    px = int(point[0] * img_w)
    py = int(point[1] * img_h)
    
    # Ensure mask is 2D
    mask_2d = mask
    if mask.ndim > 2:
        mask_2d = mask.squeeze()
        if mask_2d.ndim > 2:
            mask_2d = mask_2d[0]
    
    # Check bounds
    if px < 0 or px >= mask_2d.shape[1] or py < 0 or py >= mask_2d.shape[0]:
        return False
    
    # Check if point is on mask (value > 0.5)
    return mask_2d[py, px] > 0.5


def calculate_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x0, y0, x1, y1] format
        box2: [x0, y0, x1, y1] format
        
    Returns:
        IoU value between 0 and 1
    """
    # Ensure boxes are numpy arrays and extract scalar values
    box1 = to_numpy_array(box1)
    box2 = to_numpy_array(box2)
    
    # Extract scalar coordinates safely
    def get_scalar(arr, idx):
        val = arr[idx] if hasattr(arr, '__getitem__') else arr
        if isinstance(val, np.ndarray):
            return float(val.item() if val.size == 1 else val.flat[0])
        return float(val)
    
    x0_1 = get_scalar(box1, 0)
    y0_1 = get_scalar(box1, 1)
    x1_1 = get_scalar(box1, 2)
    y1_1 = get_scalar(box1, 3)
    
    x0_2 = get_scalar(box2, 0)
    y0_2 = get_scalar(box2, 1)
    x1_2 = get_scalar(box2, 2)
    y1_2 = get_scalar(box2, 3)
    
    # Calculate intersection
    x0_int = max(x0_1, x0_2)
    y0_int = max(y0_1, y0_2)
    x1_int = min(x1_1, x1_2)
    y1_int = min(y1_1, y1_2)
    
    if x1_int <= x0_int or y1_int <= y0_int:
        return 0.0
    
    intersection = (x1_int - x0_int) * (y1_int - y0_int)
    
    # Calculate union
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return float(intersection / union)


def calculate_tight_bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Calculate a tight bounding box that perfectly fits the mask.
    
    Args:
        mask: 2D binary numpy array (H, W) with values 0 or 1
        
    Returns:
        [x0, y0, x1, y1] bounding box in pixel coordinates
    """
    # Convert to numpy array if it's not already
    mask = to_numpy_array(mask)
    
    # Ensure mask is 2D
    if mask.ndim > 2:
        mask = mask.squeeze()
        if mask.ndim > 2:
            # Take first channel if still 3D
            mask = mask[0]
    
    # Convert to binary if needed
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    # Find rows and columns with at least one foreground pixel
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # Empty mask, return zero box
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    # Get bounding box coordinates
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    
    # Handle case where there's only one row or column
    # Ensure we extract scalar values, not arrays
    if len(row_indices) == 1:
        y0 = y1 = int(np.asarray(row_indices[0]).item())
    else:
        y0 = int(np.asarray(row_indices[0]).item())
        y1 = int(np.asarray(row_indices[-1]).item())
    
    if len(col_indices) == 1:
        x0 = x1 = int(np.asarray(col_indices[0]).item())
    else:
        x0 = int(np.asarray(col_indices[0]).item())
        x1 = int(np.asarray(col_indices[-1]).item())
    
    # Return [x0, y0, x1, y1]
    return np.array([float(x0), float(y0), float(x1 + 1), float(y1 + 1)])


def remove_mask_overlap(masks: list, boxes: list) -> list:
    """
    Remove mask overlap by mapping overlapping pixels to the mask that was there first.
    
    Args:
        masks: List of 2D numpy arrays (masks)
        boxes: List of [x0, y0, x1, y1] bounding boxes
        
    Returns:
        List of masks with overlaps removed (earlier masks take priority)
    """
    # Safely check if masks is empty (handle arrays)
    try:
        if masks is None or len(masks) == 0:
            return masks
    except (TypeError, ValueError):
        # If masks is an array, convert it
        if hasattr(masks, '__len__'):
            if len(masks) == 0:
                return masks
        return masks
    
    # Convert masks to numpy for processing (handle MLX arrays)
    masks_np = np.array(masks)
    
    # Handle 4D format [N, 1, H, W] - squeeze channel dimension if present
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0, :, :]  # Remove channel dimension: [N, H, W]
    
    # Convert first mask to numpy and get dimensions safely
    first_mask = masks_np[0]
    if first_mask.ndim > 2:
        first_mask = first_mask.squeeze()
        if first_mask.ndim > 2:
            first_mask = first_mask[0]
    
    # Get image dimensions - ensure we have integers
    shape = first_mask.shape
    # Safely extract shape dimensions as integers
    try:
        h = int(shape[0]) if len(shape) >= 1 else 1
        w = int(shape[1]) if len(shape) >= 2 else 1
    except (TypeError, ValueError):
        # If shape elements are arrays, convert them properly
        h = int(np.asarray(shape[0]).item()) if len(shape) >= 1 else 1
        w = int(np.asarray(shape[1]).item()) if len(shape) >= 2 else 1
    
    # Create a combined mask with instance IDs
    instance_mask = np.zeros((h, w), dtype=np.int32) - 1  # -1 means no instance
    
    # Process masks in order (earlier masks take priority)
    result_masks = []
    num_masks = masks_np.shape[0] if hasattr(masks_np, 'shape') else len(masks_np)
    for i in range(num_masks):
        mask = masks_np[i]
        # Ensure mask is 2D and binary
        mask_np = to_numpy_array(mask)
        if mask_np.ndim > 2:
            mask_np = mask_np.squeeze()
            if mask_np.ndim > 2:
                mask_np = mask_np[0]
        
        mask_binary = (mask_np > 0.5).astype(np.uint8)
        
        # Resize if needed to match image dimensions
        if mask_binary.shape != (h, w):
            from PIL import Image
            mask_pil = Image.fromarray((mask_binary * 255).astype(np.uint8), mode="L")
            mask_pil = mask_pil.resize((w, h), Image.BILINEAR)
            mask_binary = (np.array(mask_pil) / 255.0 > 0.5).astype(np.uint8)
        
        # Only keep pixels that aren't already assigned to an earlier instance
        new_mask = mask_binary.copy()
        new_mask[instance_mask >= 0] = 0  # Remove overlap with earlier masks
        
        # Update instance mask
        instance_mask[new_mask > 0] = i
        
        result_masks.append(new_mask.astype(np.float32))
    
    # Convert result masks back to MLX arrays for the processor
    # Processor expects masks in format [N, 1, H, W] (4D) to match new_masks format
    if result_masks:
        # Stack masks into a single array: [N, H, W]
        masks_array = np.stack(result_masks, axis=0)
        # Add channel dimension to match processor format: [N, 1, H, W]
        masks_array = masks_array[:, None, :, :]
        return mx.array(masks_array)
    else:
        return mx.zeros((0, 1, h, w))


@app.post("/update-category")
async def update_category(request: UpdateCategoryRequest):
    """Update the category_id for a specific instance."""
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        state = session["state"]
        
        # Check if we have masks
        if "masks" not in state or state["masks"] is None or len(state["masks"]) == 0:
            raise HTTPException(status_code=400, detail="No instances available")
        
        # Validate index
        num_instances = len(state["masks"])
        if request.instance_index < 0 or request.instance_index >= num_instances:
            raise HTTPException(
                status_code=400, 
                detail=f"Instance index {request.instance_index} is out of range. Valid range: 0-{num_instances-1}"
            )
        
        # Validate category_id if provided (should be 1-5 for fungus categories)
        if request.category_id is not None:
            if request.category_id < 1 or request.category_id > 5:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid category_id {request.category_id}. Valid range: 1-5 or null"
                )
        
        # Initialize category_ids if not present
        if "category_ids" not in state:
            state["category_ids"] = [None] * num_instances
        
        # Ensure category_ids list is the right length
        while len(state["category_ids"]) < num_instances:
            state["category_ids"].append(None)
        
        # Update the category_id for the specified instance
        state["category_ids"][request.instance_index] = request.category_id
        
        session["state"] = state
        
        return {
            "session_id": request.session_id,
            "message": f"Category for instance {request.instance_index} updated to {request.category_id}",
            "results": serialize_state(state),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating category: {str(e)}")


@app.post("/remove-instance")
async def remove_instance(request: RemoveInstanceRequest):
    """Remove a specific instance from the segmentation results."""
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        state = session["state"]
        
        # Check if we have masks
        if "masks" not in state or state["masks"] is None or len(state["masks"]) == 0:
            raise HTTPException(status_code=400, detail="No instances available")
        
        # Validate index
        num_instances = len(state["masks"])
        if request.instance_index < 0 or request.instance_index >= num_instances:
            raise HTTPException(
                status_code=400,
                detail=f"Instance index {request.instance_index} is out of range. Valid range: 0-{num_instances-1}"
            )
        
        # Remove the instance at the specified index
        # Handle MLX arrays for masks and boxes
        if hasattr(state["masks"], 'shape'):
            # MLX array - convert to numpy, remove index, convert back to MLX
            masks_np = np.array(state["masks"])
            indices = [i for i in range(masks_np.shape[0]) if i != request.instance_index]
            if indices:
                state["masks"] = mx.array(masks_np[indices])
            else:
                # No masks left - get dimensions from first mask
                h, w = masks_np.shape[2], masks_np.shape[3]
                state["masks"] = mx.zeros((0, 1, h, w))
        else:
            state["masks"] = [mask for i, mask in enumerate(state["masks"]) if i != request.instance_index]
        
        if "boxes" in state and state["boxes"] is not None:
            if hasattr(state["boxes"], 'shape'):
                # MLX array - convert to numpy, remove index, convert back to MLX
                boxes_np = np.array(state["boxes"])
                indices = [i for i in range(boxes_np.shape[0]) if i != request.instance_index]
                if indices:
                    state["boxes"] = mx.array(boxes_np[indices])
                else:
                    state["boxes"] = mx.zeros((0, 4))
            else:
                state["boxes"] = [box for i, box in enumerate(state["boxes"]) if i != request.instance_index]
        
        if "scores" in state and state["scores"] is not None:
            if hasattr(state["scores"], 'shape'):
                scores_np = np.array(state["scores"])
                indices = [i for i in range(scores_np.shape[0]) if i != request.instance_index]
                if indices:
                    state["scores"] = mx.array(scores_np[indices])
                else:
                    state["scores"] = mx.zeros((0,))
            else:
                state["scores"] = [score for i, score in enumerate(state["scores"]) if i != request.instance_index]
        
        if "category_ids" in state and state["category_ids"] is not None:
            state["category_ids"] = [cat_id for i, cat_id in enumerate(state["category_ids"]) if i != request.instance_index]
        
        # Remove the corresponding prompt (box or point) that created this instance
        # We assume prompts are in the same order as instances (instance i was created by prompt i)
        # When removing instance at index i, remove prompt at index i
        if "prompted_boxes" in state and state["prompted_boxes"]:
            # Check if we have enough boxes to match (if instance_index is within range)
            if request.instance_index < len(state["prompted_boxes"]):
                state["prompted_boxes"] = [box for i, box in enumerate(state["prompted_boxes"]) if i != request.instance_index]
        
        if "prompted_points" in state and state["prompted_points"]:
            # Check if we have enough points to match (if instance_index is within range)
            if request.instance_index < len(state["prompted_points"]):
                state["prompted_points"] = [point for i, point in enumerate(state["prompted_points"]) if i != request.instance_index]
        
        session["state"] = state
        
        return {
            "session_id": request.session_id,
            "message": f"Instance {request.instance_index} removed",
            "results": serialize_state(state),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing instance: {str(e)}")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free memory."""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

