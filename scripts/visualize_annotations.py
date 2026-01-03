#!/usr/bin/env python3
"""
Visualize COCO-format annotation JSON files.
Shows segmentation masks, bounding boxes, class labels, and probabilities.
Works both before classification (category_id is null) and after classification.
"""

import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple


def rle_to_mask(rle: dict) -> np.ndarray:
    """Convert RLE dict to binary mask numpy array."""
    # Try pycocotools first (more robust)
    try:
        from pycocotools import mask as mask_util
        # Convert to pycocotools format if needed
        if isinstance(rle["counts"], list):
            # Convert list to string format for pycocotools
            rle_copy = rle.copy()
            rle_copy["counts"] = " ".join(str(x) for x in rle["counts"])
            return mask_util.decode(rle_copy)
        else:
            return mask_util.decode(rle)
    except ImportError:
        pass
    
    # Fallback to custom decoder
    height, width = rle["size"]
    height = int(height)
    width = int(width)
    total_pixels = height * width
    mask = np.zeros(total_pixels, dtype=np.uint8)
    
    counts = rle["counts"]
    if isinstance(counts, str):
        # Parse space-separated string
        counts = [int(x) for x in counts.split() if x]
    elif isinstance(counts, list):
        # Already a list
        counts = [int(x) for x in counts]
    
    pixel_idx = 0
    is_foreground = False
    
    for count in counts:
        if count <= 0:
            continue
            
        if is_foreground:
            end_idx = min(pixel_idx + count, total_pixels)
            if pixel_idx < total_pixels:
                mask[pixel_idx:end_idx] = 1
            pixel_idx = end_idx
        else:
            pixel_idx += count
            
        is_foreground = not is_foreground
        
        # Safety check
        if pixel_idx >= total_pixels:
            break
    
    return mask.reshape((height, width))


def get_category_name(category_id: Optional[int], categories: List[Dict]) -> str:
    """Get category name from ID, or return 'Unlabeled' if None."""
    if category_id is None:
        return "Unlabeled"
    
    for cat in categories:
        if cat["id"] == category_id:
            return cat["name"]
    return f"Unknown (ID: {category_id})"


def get_color_for_instance(idx: int, category_id: Optional[int] = None) -> Tuple[int, int, int]:
    """Get a distinct color for each instance."""
    # Color palette for different instances
    colors = [
        (59, 235, 161),   # Emerald
        (96, 165, 250),   # Blue
        (251, 191, 36),   # Amber
        (239, 68, 68),    # Red
        (168, 85, 247),   # Purple
        (34, 197, 94),    # Green
        (249, 115, 22),   # Orange
        (236, 72, 153),   # Pink
        (14, 165, 233),   # Cyan
        (234, 179, 8),    # Yellow
    ]
    
    # Use category-based color if available, otherwise use index-based
    if category_id is not None:
        return colors[category_id % len(colors)]
    return colors[idx % len(colors)]


def visualize_annotations(
    json_path: Path,
    image_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    show_labels: bool = True,
    show_scores: bool = True,
    mask_alpha: float = 0.5
):
    """
    Visualize annotations from a COCO-format JSON file.
    
    Args:
        json_path: Path to the annotation JSON file
        image_dir: Directory containing images (default: same as JSON file)
        output_path: Path to save visualization (default: same as JSON with _vis suffix)
        show_labels: Whether to show class labels
        show_scores: Whether to show probability scores
        mask_alpha: Transparency of mask overlay (0.0 to 1.0)
    """
    # Load JSON file
    print(f"Loading annotations from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Get image info
    if len(data["images"]) == 0:
        raise ValueError("No images found in JSON file")
    
    image_info = data["images"][0]
    image_id = image_info["id"]
    image_filename = image_info["file_name"]
    image_width = int(image_info["width"])
    image_height = int(image_info["height"])
    
    # Determine image path
    if image_dir is None:
        # Try to find image relative to JSON file
        json_dir = json_path.parent
        # Check common locations
        possible_dirs = [
            json_dir,
            json_dir.parent / "assets" / "images",
            json_dir.parent.parent / "assets" / "images",
        ]
        image_path = None
        for dir_path in possible_dirs:
            candidate = dir_path / image_filename
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path is None:
            raise FileNotFoundError(
                f"Could not find image file '{image_filename}'. "
                f"Tried: {[str(d / image_filename) for d in possible_dirs]}"
            )
    else:
        image_path = image_dir / image_filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert("RGB")
    
    # Verify image dimensions match
    if image.size != (image_width, image_height):
        print(f"Warning: Image size mismatch. JSON: {image_width}x{image_height}, "
              f"Actual: {image.size[0]}x{image.size[1]}. Using actual size.")
        image_width, image_height = image.size
    
    # Get categories
    categories = data.get("categories", [])
    
    # Create visualization
    vis_image = image.copy()
    
    # Convert to RGBA for overlay
    vis_image = vis_image.convert("RGBA")
    
    # Draw each annotation
    annotations = data.get("annotations", [])
    print(f"Visualizing {len(annotations)} annotations...")
    
    for idx, ann in enumerate(annotations):
        # Skip if annotation doesn't match this image
        if ann.get("image_id") != image_id:
            continue
        
        # Get annotation data
        category_id = ann.get("category_id")
        segmentation = ann.get("segmentation")
        bbox = ann.get("bbox")  # [x0, y0, x1, y1] format
        score = ann.get("score", 0.0)
        
        # Get color for this instance
        color = get_color_for_instance(idx, category_id)
        
        # Decode and draw mask
        if segmentation:
            try:
                mask_binary = rle_to_mask(segmentation)
                
                # Resize mask if needed
                mask_h, mask_w = mask_binary.shape
                if (mask_h, mask_w) != (image_height, image_width):
                    mask_img = Image.fromarray((mask_binary * 255).astype(np.uint8), mode="L")
                    mask_img = mask_img.resize((int(image_width), int(image_height)), Image.NEAREST)
                    mask_binary = np.array(mask_img) > 127
                
                # Create mask overlay
                mask_overlay = Image.new("RGBA", (int(image_width), int(image_height)), (0, 0, 0, 0))
                mask_pixels = mask_overlay.load()
                
                for y in range(int(image_height)):
                    for x in range(int(image_width)):
                        if mask_binary[y, x] > 0:
                            mask_pixels[x, y] = (*color, int(255 * mask_alpha))
                
                # Composite mask overlay
                vis_image = Image.alpha_composite(vis_image, mask_overlay)
            except Exception as e:
                print(f"Warning: Could not decode mask for annotation {ann.get('id')}: {e}")
        
        # Draw bounding box
        if bbox:
            draw = ImageDraw.Draw(vis_image)
            x0, y0, x1, y1 = bbox
            # Ensure coordinates are integers
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            
            # Draw box outline
            draw.rectangle([x0, y0, x1, y1], outline=(*color, 255), width=3)
            
            # Prepare label text
            label_parts = []
            if show_labels:
                category_name = get_category_name(category_id, categories)
                label_parts.append(category_name)
            
            if show_scores and score is not None:
                label_parts.append(f"{score:.2f}")
            
            if label_parts:
                label_text = " | ".join(label_parts)
                
                # Try to use a nice font, fallback to default
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
                except:
                    try:
                        font = ImageFont.truetype("arial.ttf", 14)
                    except:
                        font = ImageFont.load_default()
                
                # Get text size
                bbox_text = draw.textbbox((0, 0), label_text, font=font)
                text_width = int(bbox_text[2] - bbox_text[0])
                text_height = int(bbox_text[3] - bbox_text[1])
                
                # Draw text background
                text_x = int(x0)
                text_y = int(y0 - text_height - 4)
                if text_y < 0:
                    text_y = int(y0 + 4)
                
                # Draw semi-transparent background
                text_bg = Image.new("RGBA", (int(text_width + 8), int(text_height + 4)), (*color, 200))
                vis_image.paste(text_bg, (int(text_x - 4), int(text_y - 2)), text_bg)
                
                # Draw text
                draw.text((text_x, text_y), label_text, fill=(255, 255, 255, 255), font=font)
    
    # Convert back to RGB for saving
    vis_image = vis_image.convert("RGB")
    
    # Determine output path
    if output_path is None:
        output_path = json_path.parent / f"{json_path.stem}_visualization.png"
    
    # Save visualization
    print(f"Saving visualization to {output_path}...")
    vis_image.save(output_path)
    print(f"Done! Visualization saved to {output_path}")
    
    return vis_image


def main():
    parser = argparse.ArgumentParser(
        description="Visualize COCO-format annotation JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize annotations (auto-find image):
  python visualize_annotations.py annotations/session_20251228_175838/2_annotations.json
  
  # Specify image directory:
  python visualize_annotations.py 2_annotations.json --image-dir assets/images/
  
  # Custom output path:
  python visualize_annotations.py 2_annotations.json --output my_vis.png
  
  # Hide scores:
  python visualize_annotations.py 2_annotations.json --no-scores
        """
    )
    
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to the annotation JSON file"
    )
    
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory containing images (default: auto-detect)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for visualization (default: <json_file>_visualization.png)"
    )
    
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide class labels"
    )
    
    parser.add_argument(
        "--no-scores",
        action="store_true",
        help="Hide probability scores"
    )
    
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.5,
        help="Mask overlay transparency (0.0 to 1.0, default: 0.5)"
    )
    
    args = parser.parse_args()
    
    if not args.json_file.exists():
        parser.error(f"JSON file not found: {args.json_file}")
    
    try:
        visualize_annotations(
            json_path=args.json_file,
            image_dir=args.image_dir,
            output_path=args.output,
            show_labels=not args.no_labels,
            show_scores=not args.no_scores,
            mask_alpha=args.mask_alpha
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

