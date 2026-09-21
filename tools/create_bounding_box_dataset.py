"""
create_bounding_box_dataset.py
==============================
Generates a new dataset of bounded-box crops from raw images.
It completely bypasses black background masking (mask_background=False).
Utilizes CPU multiprocessing for maximum speed.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import concurrent.futures

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from leaf_isolator import isolate_leaf
import argparse

def process_image(args):
    img_path, out_path = args
    try:
        # Skip if already exists (allows resuming)
        if os.path.exists(out_path):
            return True
            
        # Call isolator, forcefully bypass the background neutral mask
        res = isolate_leaf(img_path, target_size=(300, 300), mask_background=False)
        
        # We want the unmasked, raw bounding box crop
        crop = res['crop_raw_uint8']
        
        # Save
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img = Image.fromarray(crop)
        img.save(out_path)
        return True
    except Exception as e:
        print(f"Failed on {img_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Create Bounding Box Dataset")
    parser.add_argument("--src", default=r"clean_dataset\tomato_dataset", help="Source dataset root")
    parser.add_argument("--dst", default=r"bounding_box_dataset\tomato", help="Destination dataset root")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Number of CPU workers")
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()

    if not src_root.exists():
        print(f"Error: Source directory {src_root} not found!")
        sys.exit(1)

    print(f"Source: {src_root}")
    print(f"Target: {dst_root}")
    print(f"Workers: {args.workers}")

    # Discover all images
    tasks = []
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    
    for split in ['train', 'val', 'test']:
        split_dir = src_root / split
        if not split_dir.exists():
            continue
            
        for class_name in os.listdir(split_dir):
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
                
            out_class_dir = dst_root / split / class_name
            os.makedirs(out_class_dir, exist_ok=True)
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(valid_exts):
                    img_path = class_dir / img_name
                    out_path = out_class_dir / img_name
                    tasks.append((str(img_path), str(out_path)))

    print(f"Found {len(tasks)} images to process.")
    
    if len(tasks) == 0:
        print("No images found. Exiting.")
        return

    # Process in parallel
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # submit all
        futures = {executor.submit(process_image, task): task for task in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            success = future.result()
            if success:
                completed += 1
            
            # Simple progress print
            if completed % 100 == 0 or completed == len(tasks):
                print(f"Progress: {completed} / {len(tasks)} ({(completed/len(tasks))*100:.1f}%)")

    print("\nDataset generation complete!")

if __name__ == "__main__":
    main()
