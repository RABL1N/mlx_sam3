import json
from pathlib import Path
from PIL import Image
import subprocess
import sys


def display_image(image_path):
    """Display image using system default viewer."""
    try:
        # macOS
        subprocess.run(["open", str(image_path)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Linux
            subprocess.run(["xdg-open", str(image_path)], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Windows
            subprocess.run(["start", str(image_path)], shell=True, check=True)


def load_annotations(annotation_file):
    """Load existing annotations."""
    with open(annotation_file, "r") as f:
        return json.load(f)


def save_annotations(annotation_file, data):
    """Save annotations."""
    # Ensure categories are included
    if not data.get("categories") or len(data["categories"]) == 0:
        data["categories"] = [
            {"id": 1, "name": "aspergillus", "supercategory": "fungus"},
            {"id": 2, "name": "penicillium", "supercategory": "fungus"},
            {"id": 3, "name": "rhizopus", "supercategory": "fungus"},
            {"id": 4, "name": "mucor", "supercategory": "fungus"},
            {"id": 5, "name": "other_fungus", "supercategory": "fungus"},
        ]
    with open(annotation_file, "w") as f:
        json.dump(data, f, indent=2)


def label_instances_interactive(annotation_file, output_dir):
    """Interactive labeling of instances."""
    data = load_annotations(annotation_file)
    
    # Use categories from JSON, or default fungus categories if empty
    if not data.get("categories") or len(data["categories"]) == 0:
        default_categories = [
            {"id": 1, "name": "aspergillus", "supercategory": "fungus"},
            {"id": 2, "name": "penicillium", "supercategory": "fungus"},
            {"id": 3, "name": "rhizopus", "supercategory": "fungus"},
            {"id": 4, "name": "mucor", "supercategory": "fungus"},
            {"id": 5, "name": "other_fungus", "supercategory": "fungus"},
        ]
        data["categories"] = default_categories
    
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    categories_list = [(id, name) for id, name in sorted(categories.items())]
    
    print("\n" + "="*60)
    print("Interactive Instance Labeling")
    print("="*60)
    print(f"\nCategories:")
    for cat_id, cat_name in categories_list:
        print(f"  {cat_id}: {cat_name}")
    print(f"  0: Skip/Delete this instance")
    print(f"  q: Quit and save")
    print("="*60)
    
    annotations = data["annotations"]
    image_id = data["images"][0]["id"]
    
    for idx, ann in enumerate(annotations):
        if ann["category_id"] is not None:
            print(f"\nInstance {idx+1}/{len(annotations)}: Already labeled as '{categories[ann['category_id']]}'")
            continue
        
        instance_id = ann.get("instance_id", idx)
        vis_path = output_dir / f"{image_id}_instance_{instance_id:03d}_vis.png"
        
        if not vis_path.exists():
            print(f"\nWarning: Visualization not found: {vis_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Instance {idx+1}/{len(annotations)}")
        print(f"Score: {ann['score']:.3f}")
        print(f"BBox: {ann['bbox']}")
        print(f"{'='*60}")
        
        # Display the visualization
        print(f"\nOpening visualization: {vis_path.name}")
        display_image(vis_path)
        
        # Get user input
        while True:
            try:
                choice = input(f"\nSelect category (1-{len(categories)}, 0=skip, q=quit): ").strip().lower()
                
                if choice == 'q':
                    print("\nSaving and exiting...")
                    save_annotations(annotation_file, data)
                    return
                
                if choice == '0':
                    print("Skipping this instance...")
                    break
                
                cat_id = int(choice)
                if cat_id in categories:
                    ann["category_id"] = cat_id
                    print(f"✓ Labeled as '{categories[cat_id]}'")
                    # Auto-save after each label
                    save_annotations(annotation_file, data)
                    break
                else:
                    print(f"Invalid category ID. Choose 1-{len(categories)}, 0, or q")
            except ValueError:
                print("Invalid input. Enter a number or 'q'")
            except KeyboardInterrupt:
                print("\n\nInterrupted. Saving progress...")
                save_annotations(annotation_file, data)
                sys.exit(0)
    
    print(f"\n{'='*60}")
    print("Labeling complete!")
    print(f"{'='*60}")
    
    # Final save
    save_annotations(annotation_file, data)
    
    # Summary
    labeled = sum(1 for ann in annotations if ann["category_id"] is not None)
    print(f"\nSummary: {labeled}/{len(annotations)} instances labeled")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive instance labeling")
    parser.add_argument("--annotation-file", type=str, default="annotations/1_annotations.json",
                       help="Path to annotation JSON file")
    parser.add_argument("--output-dir", type=str, default="annotations",
                       help="Directory containing visualization images")
    
    args = parser.parse_args()
    
    annotation_file = Path(args.annotation_file)
    output_dir = Path(args.output_dir)
    
    if not annotation_file.exists():
        print(f"Error: Annotation file not found: {annotation_file}")
        print("Run annotate.py first to generate annotations.")
        sys.exit(1)
    
    label_instances_interactive(annotation_file, output_dir)


if __name__ == "__main__":
    main()

