"""
leaf_isolator.py (Backward-Compatibility Shim)
==============================================
The canonical LeafIsolator implementation has moved to:
  src.preprocessing.leaf_isolator
This shim preserves 100% backward compatibility for root execution.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.leaf_isolator import *

if __name__ == "__main__":
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else "test_images/potato/potatotest.png"
    if Path(test_img).exists():
        res = isolate_leaf(test_img)
        print(f"Engine Used: {res['engine_used']}")
