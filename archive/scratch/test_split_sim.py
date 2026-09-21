import pandas as pd
import numpy as np
import imagehash
from collections import Counter

df = pd.read_csv('manifests/potato/potato_pre_audit_catalog.csv')

def cluster_and_split(thresh=4):
    df_copy = df.copy()
    df_copy["group_id"] = ""
    group_counter = 0

    for cls in df_copy["class_label"].unique():
        cls_df = df_copy[df_copy["class_label"] == cls]
        indices = cls_df.index.tolist()
        hashes = [imagehash.hex_to_hash(h) for h in cls_df["phash"].tolist()]
        n = len(indices)

        parent = list(range(n))
        def find(i):
            path = []
            while parent[i] != i:
                path.append(i)
                i = parent[i]
            for node in path:
                parent[node] = i
            return i
        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        bool_hashes = np.array([h.hash.flatten() for h in hashes], dtype=np.bool_)
        for i in range(n):
            if i + 1 < n:
                dists = np.count_nonzero(bool_hashes[i] != bool_hashes[i + 1:], axis=1)
                matches = np.where(dists <= thresh)[0]
                for m in matches:
                    union(i, i + 1 + m)

        cluster_map = {}
        for i in range(n):
            root = find(i)
            if root not in cluster_map:
                group_counter += 1
                cluster_map[root] = f"potato__{cls}__grp_{group_counter:05d}"
            df_copy.loc[indices[i], "group_id"] = cluster_map[root]

    # Split
    np.random.seed(42)
    df_copy["partition"] = ""
    target_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}

    for cls in ["early_blight", "healthy", "late_blight"]:
        cls_df = df_copy[df_copy["class_label"] == cls]
        group_sizes = cls_df.groupby("group_id").size().to_dict()
        groups = list(group_sizes.keys())
        np.random.shuffle(groups)

        total_samples = len(cls_df)
        train_target = int(total_samples * target_ratios["train"])
        val_target = int(total_samples * target_ratios["val"])

        curr_train, curr_val, curr_test = 0, 0, 0
        group_partitions = {}

        # Better multi-bucket allocation (minimize deviation from target proportions)
        for g in groups:
            sz = group_sizes[g]
            # Calculate current ratios or deficiencies
            need_train = max(0, train_target - curr_train)
            need_val = max(0, val_target - curr_val)
            need_test = max(0, (total_samples - train_target - val_target) - curr_test)

            # Assign to bucket with largest relative deficit
            deficits = {
                "train": need_train / train_target if train_target > 0 else 0,
                "val": need_val / val_target if val_target > 0 else 0,
                "test": need_test / (total_samples - train_target - val_target) if (total_samples - train_target - val_target) > 0 else 0
            }
            best_part = max(deficits, key=deficits.get)
            group_partitions[g] = best_part
            if best_part == "train":
                curr_train += sz
            elif best_part == "val":
                curr_val += sz
            else:
                curr_test += sz

        for g, part in group_partitions.items():
            mask = (df_copy["class_label"] == cls) & (df_copy["group_id"] == g)
            df_copy.loc[mask, "partition"] = part

    ct = pd.crosstab(df_copy["class_label"], df_copy["partition"], margins=True)
    print("\n--- Cross-tabulation (Threshold =", thresh, ") ---")
    print(ct)

cluster_and_split(thresh=4)
