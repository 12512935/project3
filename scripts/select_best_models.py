# -*- coding: utf-8 -*-
"""
Select and save best models near target R2=0.4 (if any >0.4 choose highest) per dataset stem.
Scans data/model_objects for meta JSONs, summary CSVs, and joblib models.
Saves chosen models into data/model_objects/best_models/ and writes a summary CSV/JSON.
"""
from pathlib import Path
import json, shutil
import joblib
import re

BASE = Path(__file__).resolve().parents[1]
MO_DIR = BASE / 'data' / 'model_objects'
BEST_DIR = MO_DIR / 'best_models'
BEST_DIR.mkdir(parents=True, exist_ok=True)

# helper: load JSON if possible
def try_load_json(p):
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None

# collect candidate R2s from various files
candidates = {}  # stem -> list of dicts {model_file, r2, source, meta}

# 1) read model_wide_deep_summary.csv if exists
csvp = MO_DIR / 'model_wide_deep_summary.csv'
if csvp.exists():
    import pandas as pd
    df = pd.read_csv(csvp, encoding='utf-8-sig')
    for _, row in df.iterrows():
        stem = Path(row['file']).stem if 'file' in row and not pd.isna(row['file']) else None
        if not stem:
            continue
        r2 = None
        try:
            r2 = float(row.get('val_r2_meta', row.get('val_r2_meta', float('nan'))))
        except Exception:
            r2 = None
        if stem not in candidates:
            candidates[stem] = []
        candidates[stem].append({'model_file': row.get('model_file'), 'r2': r2, 'source': 'model_wide_deep_summary.csv', 'meta': row.to_dict()})

# 2) read any meta_*.json files
for p in MO_DIR.glob('meta_*.json'):
    j = try_load_json(p)
    if not j:
        continue
    # try to get file stem
    file_name = j.get('file') or p.stem
    stem = Path(file_name).stem
    r2 = None
    # common locations: j['metrics']['meta']['r2'] or j['metrics']['meta_r2'] or j.get('val_r2_meta')
    try:
        if 'metrics' in j and isinstance(j['metrics'], dict) and 'meta' in j['metrics']:
            r2 = float(j['metrics']['meta'].get('r2', j['metrics']['meta'].get('r2', None)))
    except Exception:
        r2 = None
    if r2 is None:
        r2 = j.get('val_r2_meta') or j.get('val_r2') or j.get('r2')
        try:
            r2 = float(r2)
        except Exception:
            r2 = None
    if stem not in candidates:
        candidates[stem] = []
    candidates[stem].append({'model_file': j.get('model_file') or p.name, 'r2': r2, 'source': str(p.name), 'meta': j})

# 3) read any *_summary.json or *_summary.csv in model_objects
for p in MO_DIR.glob('*summary*.json'):
    if p.name == 'merge_statistic_summary.json' or p.name.endswith('best.json'):
        continue
    j = try_load_json(p)
    if not j:
        continue
    # j may be list or dict
    if isinstance(j, list):
        for entry in j:
            fname = entry.get('file') or entry.get('name')
            if not fname:
                continue
            stem = Path(fname).stem
            r2 = entry.get('r2') or entry.get('val_r2') or entry.get('val_r2_meta')
            try:
                r2 = float(r2)
            except Exception:
                r2 = None
            candidates.setdefault(stem, []).append({'model_file': entry.get('model') or entry.get('model_file'), 'r2': r2, 'source': p.name, 'meta': entry})
    elif isinstance(j, dict):
        fname = j.get('file')
        if fname:
            stem = Path(fname).stem
            r2 = j.get('r2') or j.get('val_r2') or j.get('val_r2_meta')
            try:
                r2 = float(r2)
            except Exception:
                r2 = None
            candidates.setdefault(stem, []).append({'model_file': j.get('model_file'), 'r2': r2, 'source': p.name, 'meta': j})

# 4) check joblib files: try to load and inspect contained meta or metrics keys
for p in MO_DIR.glob('*.joblib'):
    name = p.name
    # infer stem from name: look for known suffixes
    m = re.match(r'.*_(?P<stem>[^_]+)\.joblib$', name)
    stem = None
    if m:
        stem = m.group('stem')
    else:
        # fallback: if name contains a file stem used in data files
        for datafile in ['HLS','JWX','JX','RZP','YJC','YJP','YJX','YJZ','YXH','YXY','ZXJZ']:
            if datafile.lower() in name.lower():
                stem = datafile
                break
    # try to load joblib and inspect
    meta_info = None
    r2 = None
    try:
        obj = joblib.load(p)
        # obj might be dict with 'meta' or 'metrics' or 'models'
        if isinstance(obj, dict):
            # wide_deep model saved earlier has 'models' key sometimes
            if 'models' in obj:
                # no metrics stored
                pass
            if 'meta' in obj:
                meta_info = obj['meta']
                r2 = meta_info.get('r2') or meta_info.get('val_r2')
            # some saved objects from other scripts contain 'model' and 'features' only
            # check for nested classifiers with attributes
        # else if obj has attribute 'feature_names_in_' or 'n_features_in_' ignore
    except Exception:
        pass
    if stem:
        candidates.setdefault(stem, []).append({'model_file': p.name, 'r2': r2, 'source': 'joblib_file', 'meta': meta_info})

# Now for each stem choose best model per user rule
BEST_SUMMARY = []
for stem, lst in sorted(candidates.items()):
    # filter out entries without r2 if possible
    processed = []
    for e in lst:
        try:
            r = e.get('r2')
            if r is None:
                r = None
            else:
                r = float(r)
        except Exception:
            r = None
        processed.append({'model_file': e.get('model_file'), 'r2': r, 'source': e.get('source'), 'meta': e.get('meta')})
    # remove duplicates by model_file
    uniq = {}
    for e in processed:
        key = str(e['model_file'])
        if key not in uniq or (e['r2'] is not None and (uniq[key]['r2'] is None or e['r2']>uniq[key]['r2'])):
            uniq[key]=e
    processed = list(uniq.values())
    # if none have r2 values, skip
    have = [e for e in processed if e['r2'] is not None]
    chosen = None
    if have:
        # if any >0.4 choose the highest
        above = [e for e in have if e['r2']>0.4]
        if above:
            chosen = max(above, key=lambda x: x['r2'])
        else:
            # choose one with r2 closest to 0.4
            chosen = min(have, key=lambda x: abs(x['r2']-0.4))
    else:
        # choose any candidate with model_file present
        if processed:
            chosen = processed[0]
    if chosen:
        # attempt to copy model file if exists
        model_src = MO_DIR / (chosen['model_file'] if chosen['model_file'] else '')
        model_dst = None
        copied = False
        if chosen.get('model_file') and model_src.exists():
            model_dst = BEST_DIR / model_src.name
            try:
                shutil.copy2(model_src, model_dst)
                copied = True
            except Exception:
                copied = False
        BEST_SUMMARY.append({'stem': stem, 'chosen_model_file': chosen.get('model_file'), 'r2': chosen.get('r2'), 'source': chosen.get('source'), 'copied': copied})

# write output summary
out_csv = MO_DIR / 'best_models_summary.csv'
import csv
with out_csv.open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['stem','chosen_model_file','r2','source','copied'])
    writer.writeheader()
    for r in BEST_SUMMARY:
        writer.writerow(r)

(MO_DIR / 'best_models_summary.json').write_text(json.dumps(BEST_SUMMARY, ensure_ascii=False, indent=2), encoding='utf-8')
print('Saved best models summary to', out_csv)

