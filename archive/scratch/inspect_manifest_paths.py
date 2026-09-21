import pandas as pd
from pathlib import Path

df = pd.read_csv("manifests/rice/split_manifest_v1.csv")
df_rice = df[df["crop"] == "rice"]
print("Total rice samples:", len(df_rice))
paths = df_rice["original_path"].tolist()

exists_count = sum(Path(p).exists() for p in paths)
print(f"Paths existing directly as written: {exists_count} / {len(paths)}")

sample_non_existent = [p for p in paths if not Path(p).exists()]
print("Non-existent count:", len(sample_non_existent))

for p in sample_non_existent[:10]:
    print("Non-existent sample:", p)
    fname = Path(p).name
    # Search for this filename in the repo!
    found = list(Path(".").rglob(fname))
    print(f"  Searching for {fname}: found -> {found}")

# Also check clean_dataset vs finaldataset
print("\nChecking clean_dataset and finaldataset directories:")
for d in ["clean_dataset", "finaldataset", "Rice___Healthy"]:
    p = Path(d)
    print(f"  {d}: exists={p.exists()}")
