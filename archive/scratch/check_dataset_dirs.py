import os
from pathlib import Path

for d in ["clean_dataset/rice_dataset", "finaldataset/rice_dataset"]:
    root = Path(d)
    print(f"\nChecking directory: {d} (exists={root.exists()})")
    if root.exists():
        for split in ["train", "val", "test"]:
            s_dir = root / split
            if s_dir.exists():
                classes = {c.name: len(list(c.glob("*.*"))) for c in s_dir.iterdir() if c.is_dir()}
                print(f"  {split} classes: {classes} (Total: {sum(classes.values())})")
