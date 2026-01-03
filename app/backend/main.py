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
            mask_np = np.array(masks[i])
            box_np = np.array(boxes[i])
            score_np = float(np.array(scores[i]))
            
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
        result["scores"] = scores_list
    
    if "prompted_boxes" in state:
        result["prompted_boxes"] = state["prompted_boxes"]
    
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
        
        # Store prompted box for display
        if "prompted_boxes" not in state:
            state["prompted_boxes"] = []
        
        # Convert from normalized cxcywh to pixel xyxy for display
        img_w = state["original_width"]
        img_h = state["original_height"]
        cx, cy, w, h = request.box
        x_min = (cx - w / 2) * img_w
        y_min = (cy - h / 2) * img_h
        x_max = (cx + w / 2) * img_w
        y_max = (cy + h / 2) * img_h
        
        state["prompted_boxes"].append({
            "box": [x_min, y_min, x_max, y_max],
            "label": request.label
        })
        
        start_time = time.perf_counter()
        state = processor.add_geometric_prompt(request.box, request.label, state)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        session["state"] = state
        
        return {
            "session_id": request.session_id,
            "box_type": "positive" if request.label else "negative",
            "results": serialize_state(state),
            "processing_time_ms": round(processing_time_ms, 2)
        }
    
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
        
        start_time = time.perf_counter()
        processor.reset_all_prompts(state)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        if "prompted_boxes" in state:
            del state["prompted_boxes"]
        
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
        
        start_time = time.perf_counter()
        state = processor.add_point_prompt(request.point, request.label, state)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        session["state"] = state
        
        return {
            "session_id": request.session_id,
            "point_type": "positive" if request.label else "negative",
            "results": serialize_state(state),
            "processing_time_ms": round(processing_time_ms, 2)
        }
    
    except Exception as e:
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
                    "category_id": None,
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

