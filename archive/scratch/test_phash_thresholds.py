import pandas as pd
import numpy as np
import imagehash
from collections import Counter

df = pd.read_csv('manifests/potato/potato_pre_audit_catalog.csv')
print('Total rows:', len(df))

for cls in ['early_blight', 'healthy', 'late_blight']:
    cls_df = df[df['class_label'] == cls]
    hashes = [imagehash.hex_to_hash(h) for h in cls_df['phash'].tolist()]
    n = len(hashes)
    bool_hashes = np.array([h.hash.flatten() for h in hashes], dtype=np.bool_)
    
    print(f"\n--- Class: {cls} (n={n}) ---")
    for thresh in [2, 3, 4, 5, 6, 8, 10]:
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
        for i in range(n):
            if i + 1 < n:
                dists = np.count_nonzero(bool_hashes[i] != bool_hashes[i + 1:], axis=1)
                matches = np.where(dists <= thresh)[0]
                for m in matches:
                    union(i, i + 1 + m)
        roots = [find(i) for i in range(n)]
        counts = Counter(roots)
        max_size = max(counts.values())
        n_clusters = len(counts)
        singletons = sum(1 for v in counts.values() if v == 1)
        large_clusters = [v for v in counts.values() if v > 20]
        print(f"Thresh {thresh:2d} | Clusters: {n_clusters:4d} | Max Size: {max_size:4d} | Singletons: {singletons:4d} | Clusters > 20: {len(large_clusters)} (sizes: {large_clusters[:5]})")
