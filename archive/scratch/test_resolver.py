import pandas as pd
from pathlib import Path

df = pd.read_csv("manifests/rice/split_manifest_v1.csv")
df_rice = df[df["crop"] == "rice"]
root_dir = Path(".")

def resolve(p_str, part, label):
    fname = Path(p_str).name
    cand = root_dir / "clean_dataset" / "rice_dataset" / part / label / fname
    if cand.exists():
        return cand
    if Path(p_str).exists():
        return Path(p_str)
    raise FileNotFoundError(p_str)

resolved = [resolve(r["original_path"], r["partition"], r["class_label"]) for _, r in df_rice.iterrows()]
print(f"Successfully resolved: {len(resolved)} / {len(df_rice)} rice images!")
