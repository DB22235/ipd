import pandas as pd
from pathlib import Path

df = pd.read_csv("manifests/rice/split_manifest_v1.csv")
df_rice = df[df["crop"] == "rice"]
print("Total rice rows in manifest:", len(df_rice))

clean_root = Path("clean_dataset/rice_dataset")
matched = 0
missing = []

for idx, row in df_rice.iterrows():
    orig_p = Path(row["original_path"])
    fname = orig_p.name
    part = row["partition"]
    label = row["class_label"]
    
    target_p = clean_root / part / label / fname
    if target_p.exists():
        matched += 1
    else:
        missing.append((row["image_id"], str(target_p), str(orig_p)))

print(f"Matched directly in clean_dataset/rice_dataset/{{partition}}/{{class_label}}/{{fname}}: {matched} / {len(df_rice)}")
if missing:
    print(f"Missing count: {len(missing)}")
    print("First 5 missing:", missing[:5])
else:
    print("[SUCCESS] 100% of all 4,932 rice images exist in clean_dataset/rice_dataset/{partition}/{class_label}/{fname}!")
