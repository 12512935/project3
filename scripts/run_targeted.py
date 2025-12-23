# -*- coding: utf-8 -*-
"""
Run targeted training using train_one from wide_deep_train.py with custom candidate MLP configs.
Stops when a file achieves val R2 >= target_r2.
"""
import sys
from pathlib import Path
import importlib.util

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / 'wide_deep_train.py'

spec = importlib.util.spec_from_file_location('wide_deep_train', MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# define files to try (best-performing candidates first)
files_to_try = ['YXY.csv', 'YXH.csv', 'JWX.csv', 'ZXJZ.csv', 'HLS.csv']
# candidate MLP architectures to try (in order)
candidates = [
    (1024, 512, 256, 128),
    (512, 256, 128, 64),
    (2048, 1024, 512, 256),
]

TARGET_R2 = 0.5

results = []
for fname in files_to_try:
    path = SCRIPT_DIR.parent / 'data' / 'cleaned_data' / fname
    if not path.exists():
        print('file not found, skipping:', fname)
        continue
    print('\n=== Trying file:', fname, '===')
    try:
        summary = mod.train_one(path, hidden_layers=candidates[0], test_size=0.2, random_state=42, max_iter=500, candidates=candidates, target_r2=TARGET_R2, use_poly=True)
        if summary:
            print('Summary:', summary)
            results.append(summary)
            if summary.get('val_r2_meta', 0) >= TARGET_R2:
                print('Target reached for', fname)
                break
    except Exception as e:
        print('Error processing', fname, e)

# write results to model_objects
import json
out_path = SCRIPT_DIR.parent / 'data' / 'model_objects' / 'targeted_run_summary.json'
out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
print('\nDone. Results saved to', out_path)

