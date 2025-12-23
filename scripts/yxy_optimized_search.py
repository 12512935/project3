# -*- coding: utf-8 -*-
"""
Targeted randomized search for YXY.csv using HGB and RF with limited iterations.
Saves best model if validation R2 >= 0.5.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import importlib.util

BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / 'scripts'
spec = importlib.util.spec_from_file_location('adv', SCRIPTS / 'advanced_train.py')
adv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adv)

CLEAN_DIR = BASE / 'data' / 'cleaned_data'
MO_DIR = BASE / 'data' / 'model_objects'
MO_DIR.mkdir(parents=True, exist_ok=True)

PATH = CLEAN_DIR / 'YXY.csv'
print('Reading', PATH)
df = pd.read_csv(PATH, encoding='utf-8-sig')
X, y = adv.build_features(df, use_poly=True)
mask = ~np.isnan(X.to_numpy()).any(axis=1)
X = X.loc[mask]; y = y[mask]
# remove outliers
ym = y.mean(); ys = y.std(); keep = (y>=ym-3*ys)&(y<=ym+3*ys)
X = X.loc[keep]; y = y[keep]

# split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

# try transforms
for tname in ('none','log1p'):
    if tname=='none':
        yt = y_train.copy(); yv = y_val.copy()
    else:
        yt = np.log1p(y_train); yv = np.log1p(y_val)

    # RandomForest random search
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_dist = {'n_estimators': [200, 500, 800], 'max_depth': [10,20,30,None], 'min_samples_split': [2,5,10]}
    print('Starting RF RandomizedSearchCV (n_iter=12) for transform', tname)
    rs_rf = RandomizedSearchCV(rf, rf_dist, n_iter=12, cv=3, random_state=42, n_jobs=-1)
    rs_rf.fit(X_train, yt)
    best_rf = rs_rf.best_estimator_
    pred_rf = best_rf.predict(X_val)
    if tname=='log1p':
        pred_rf = np.expm1(pred_rf); yv_orig = np.expm1(yv)
    else:
        yv_orig = yv
    mse_rf = mean_squared_error(yv_orig, pred_rf); rmse_rf = np.sqrt(mse_rf); r2_rf = r2_score(yv_orig, pred_rf)
    print('RF best params:', rs_rf.best_params_, 'R2=', r2_rf)
    results.append({'transform': tname, 'model': 'rf', 'r2': float(r2_rf), 'rmse': float(rmse_rf), 'params': rs_rf.best_params_})
    if r2_rf >= 0.5:
        joblib.dump({'model': best_rf, 'features': list(X.columns), 'transform': tname}, MO_DIR / 'yxy_rf_best.joblib')
        print('Saved RF model achieving R2>=0.5')
        break

    # HGB random search
    hgb = HistGradientBoostingRegressor(random_state=42)
    hgb_dist = {'max_iter': [100,200,300], 'learning_rate': [0.01,0.05,0.1], 'max_leaf_nodes': [31,63,127]}
    print('Starting HGB RandomizedSearchCV (n_iter=12) for transform', tname)
    rs_hgb = RandomizedSearchCV(hgb, hgb_dist, n_iter=12, cv=3, random_state=42, n_jobs=-1)
    rs_hgb.fit(X_train, yt)
    best_hgb = rs_hgb.best_estimator_
    pred_hgb = best_hgb.predict(X_val)
    if tname=='log1p':
        pred_hgb = np.expm1(pred_hgb); yv_orig = np.expm1(yv)
    else:
        yv_orig = yv
    mse_hgb = mean_squared_error(yv_orig, pred_hgb); rmse_hgb = np.sqrt(mse_hgb); r2_hgb = r2_score(yv_orig, pred_hgb)
    print('HGB best params:', rs_hgb.best_params_, 'R2=', r2_hgb)
    results.append({'transform': tname, 'model': 'hgb', 'r2': float(r2_hgb), 'rmse': float(rmse_hgb), 'params': rs_hgb.best_params_})
    if r2_hgb >= 0.5:
        joblib.dump({'model': best_hgb, 'features': list(X.columns), 'transform': tname}, MO_DIR / 'yxy_hgb_best.joblib')
        print('Saved HGB model achieving R2>=0.5')
        break

# save results
(MO_DIR / 'yxy_opt_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
print('Done. Results:', results)

