# -*- coding: utf-8 -*-
"""
Quick, lightweight advanced training runner to test whether stacking can reach target R2 quickly.
Uses the same feature builder as advanced_train.build_features by importing it.
Runs smaller models and optional subsampling to be fast.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json

BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / 'scripts'
import importlib.util
spec = importlib.util.spec_from_file_location('advanced_train', SCRIPTS / 'advanced_train.py')
adv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adv)

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

CLEAN_DIR = BASE / 'data' / 'cleaned_data'
MO_DIR = BASE / 'data' / 'model_objects'
MO_DIR.mkdir(parents=True, exist_ok=True)

FILES = ['YXY.csv','YXH.csv','JWX.csv','ZXJZ.csv']
TARGET_R2 = 0.5


def metrics(y_true,y_pred):
    mse = mean_squared_error(y_true,y_pred)
    return mse, float(np.sqrt(mse)), float(r2_score(y_true,y_pred))


def run_quick(fname, sample_frac=0.5, random_state=42):
    path = CLEAN_DIR / fname
    if not path.exists():
        print('file not found', fname); return None
    print('Running quick on', fname)
    df = pd.read_csv(path, encoding='utf-8-sig')
    # reuse feature builder
    X, y = adv.build_features(df, use_poly=True)
    # drop NaNs
    mask = ~np.isnan(X.to_numpy()).any(axis=1)
    X = X.loc[mask]; y = y[mask]
    # optional subsample
    if 0 < sample_frac < 1.0:
        X, _, y, _ = train_test_split(X, y, train_size=sample_frac, random_state=random_state)
    # remove outliers
    y_mean = y.mean(); y_std = y.std(); keep = (y>= y_mean-3*y_std) & (y<= y_mean+3*y_std)
    X = X.loc[keep]; y = y[keep]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # small models
    hgb = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_leaf_nodes=31, random_state=random_state)
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=random_state)
    ridge = Ridge(alpha=1.0)
    estimators = [('hgb', hgb), ('rf', rf), ('ridge', Pipeline([('scaler', StandardScaler()), ('ridge', ridge)]))]
    stack = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), passthrough=True, n_jobs=-1)

    # try transforms: none and log1p
    best=None
    for tname in ('none','log1p'):
        if tname=='none':
            yt = y_train.copy(); yvt = y_val.copy()
        else:
            yt = np.log1p(y_train); yvt = np.log1p(y_val)
        try:
            stack.fit(X_train.values, yt)
            pred = stack.predict(X_val.values)
            if tname=='log1p':
                pred = np.expm1(pred); yvt_orig = np.expm1(yvt)
            else:
                yvt_orig = yvt
            mse, rmse, r2 = metrics(yvt_orig, pred)
            print('  transform', tname, 'R2=', r2, 'RMSE=', rmse)
            if best is None or r2 > best[1]:
                best = (tname, r2, rmse, stack)
        except Exception as e:
            print('  failed transform', tname, e)
    if best:
        tname,r2,rmse,model = best
        print('Best for', fname, 'transform', tname, 'R2=', r2)
        if r2 >= TARGET_R2:
            joblib.dump({'model': model, 'features': list(X.columns)}, MO_DIR / f'quick_stack_{Path(fname).stem}.joblib')
            print('Saved quick_stack model for', fname)
    return {'file': fname, 'best_r2': best[1] if best else None, 'best_rmse': best[2] if best else None}


if __name__=='__main__':
    results=[]
    for f in FILES:
        r = run_quick(f, sample_frac=0.5)
        results.append(r)
    (MO_DIR / 'quick_advanced_summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Done. Summary written to', MO_DIR / 'quick_advanced_summary.json')

