import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).resolve().parent
model_path = ROOT / "models" / "tomato_teacher" / "tomato_teacher_efficientnetb3.keras"
model = tf.keras.models.load_model(str(model_path), compile=False)
classes = ["early_blight", "healthy", "late_blight"]

print("=" * 70)
print("TOMATO MODEL DIAGNOSTIC PROBE")
print("=" * 70)

# 1. Test genuine PlantVillage test image
pv_healthy = ROOT / "clean_dataset" / "tomato_dataset" / "test" / "healthy" / "94957.jpg"
if pv_healthy.exists():
    img_pv = Image.open(pv_healthy).convert("RGB").resize((300, 300))
    arr_pv = np.expand_dims(np.array(img_pv, dtype=np.float32), 0)
    pred_pv = tf.nn.softmax(model(arr_pv, training=False)).numpy()[0]
    print(f"1. PlantVillage Ground Truth Healthy (94957.jpg):")
    for c, p in zip(classes, pred_pv):
        print(f"   {c:<15}: {p*100:.2f}%")

# 2. Test tomatotest10 across background fill levels
from leaf_isolator import isolate_leaf
print("\n2. tomatotest10.webp across background fill levels:")
for bg_val in [114, 50, 0]:
    iso = isolate_leaf(ROOT / "tomatotest10.webp", mask_background=True, bg_neutral_val=bg_val)
    crop = iso["cropped_image"]
    pred = tf.nn.softmax(model(np.expand_dims(crop, 0), training=False)).numpy()[0]
    top_c = classes[np.argmax(pred)]
    top_p = np.max(pred) * 100
    print(f"   Fill val {bg_val:>3}: Diagnosed {top_c.upper():<12} ({top_p:.1f}%) | Probs: {[round(float(p)*100, 1) for p in pred]}")

# 3. Test tomatotest7 across background fill levels
print("\n3. tomatotest7.webp across background fill levels:")
for bg_val in [114, 50, 0]:
    iso7 = isolate_leaf(ROOT / "tomatotest7.webp", mask_background=True, bg_neutral_val=bg_val)
    crop7 = iso7["cropped_image"]
    pred7 = tf.nn.softmax(model(np.expand_dims(crop7, 0), training=False)).numpy()[0]
    top_c7 = classes[np.argmax(pred7)]
    top_p7 = np.max(pred7) * 100
    print(f"   Fill val {bg_val:>3}: Diagnosed {top_c7.upper():<12} ({top_p7:.1f}%) | Probs: {[round(float(p)*100, 1) for p in pred7]}")

print("=" * 70)
