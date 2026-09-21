import glob, os
from collections import Counter

patterns = ['**/*potato*.*', '**/*Potato*.*']
found = set()
for pat in patterns:
    for f in glob.glob(pat, recursive=True):
        ext = os.path.splitext(f)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.webp'] and not f.startswith('.'):
            found.add(f)
print(f'Total potato images found: {len(found)}')
folders = Counter(os.path.dirname(f) for f in found)
for k, v in sorted(folders.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f'  {k}: {v}')
