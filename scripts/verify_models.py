# -*- coding: utf-8 -*-
"""
验证 data/model_objects 中每个模型：
- 读取 model_<stem>.joblib
- 读取 val_<stem>.csv
- 计算 val_mse 和 val_r2，并与 meta_<stem>.json 中记录的值对比

用法：python scripts\verify_models.py
"""
from pathlib import Path
import json
import sys

try:
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score
except Exception as e:
    print('缺少依赖，请安装 pandas scikit-learn joblib')
    raise

BASE = Path(__file__).resolve().parents[1]
MO_DIR = BASE / 'data' / 'model_objects'

models = sorted([p for p in MO_DIR.iterdir() if p.name.startswith('model_') and p.suffix=='.joblib'])

if not models:
    print('未找到任何已保存的模型。')
    sys.exit(0)

for mpath in models:
    stem = mpath.stem.replace('model_','')
    meta_path = MO_DIR / f'meta_{stem}.json'
    val_path = MO_DIR / f'val_{stem}.csv'
    print('---', stem)
    if not meta_path.exists():
        print('  警告: 未找到 meta 文件', meta_path)
        continue
    if not val_path.exists():
        print('  警告: 未找到验证集文件', val_path)
        continue
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    try:
        model = joblib.load(mpath)
    except Exception as e:
        print('  无法加载模型:', e)
        continue
    try:
        df_val = pd.read_csv(val_path, encoding='utf-8-sig')
    except Exception as e:
        print('  无法读取验证集:', e)
        continue
    if not {'M','C','d','L','R'}.issubset(df_val.columns):
        print('  验证集缺少列')
        continue
    X_val = df_val[['M','C','d','L']].astype(float).values
    y_val = df_val['R'].astype(float).values
    y_pred = model.predict(X_val)
    mse = float(mean_squared_error(y_val, y_pred))
    r2 = float(r2_score(y_val, y_pred))
    print(f"  meta_val_mse={meta.get('val_mse')}, actual_val_mse={mse}")
    print(f"  meta_val_r2={meta.get('val_r2')}, actual_val_r2={r2}")

