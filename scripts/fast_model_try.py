# -*- coding: utf-8 -*-
"""
Fast model try: try Ridge, RandomForest, HistGradientBoosting with wide+deep meta on specified files.
Saves best model info to data/model_objects/fast_try_summary.json and joblib for model achieving target.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

BASE = Path(__file__).resolve().parents[1]
CLEAN_DIR = BASE / 'data' / 'cleaned_data'
MO_DIR = BASE / 'data' / 'model_objects'
MO_DIR.mkdir(parents=True, exist_ok=True)

FILES = ['YXY.csv','YXH.csv','JWX.csv','ZXJZ.csv']
TARGET_R2 = 0.5


def build_features(df, use_poly=False):
    required = ['M','C','d','L','R']
    df = df.copy()
    df = df.dropna(subset=required)
    X = pd.DataFrame()
    X['M'] = df['M'].astype(float)
    X['C'] = df['C'].astype(float)
    X['d'] = df['d'].astype(float)
    X['L'] = df['L'].astype(float)
    for extra in ['Max_Ve','Min_Ve','Avg_Ve']:
        if extra in df.columns:
            X[extra] = pd.to_numeric(df[extra], errors='coerce').fillna(0.0)
    X['M_over_L'] = (X['M'] / X['L'].replace(0, np.nan)).fillna(0.0)
    X['C2'] = X['C']**2
    y = df['R'].astype(float).values
    if use_poly:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        Xp = pd.DataFrame(poly.fit_transform(X.values), columns=poly.get_feature_names_out(X.columns), index=X.index)
        return Xp, y
    return X, y


def metrics(y_true,y_pred):
    mse = mean_squared_error(y_true,y_pred)
    return mse, float(np.sqrt(mse)), float(r2_score(y_true,y_pred))


summary = []

for fname in FILES:
    path = CLEAN_DIR / fname
    if not path.exists():
        print('skip missing', fname)
        continue
    print('\n=== File:', fname, '===')
    df = pd.read_csv(path, encoding='utf-8-sig')
    # try with and without poly and with log-target
    tried_records = []
    for use_poly in (False, True):
        X, y = build_features(df, use_poly=use_poly)
        # remove rows with NaN in X
        mask = ~np.isnan(X.to_numpy()).any(axis=1)
        X = X.loc[mask]
        y = y[mask]
        # try log1p transform of target
        for logt in (False, True):
            y_use = np.log1p(y) if logt else y
            X_train, X_val, y_train, y_val = train_test_split(X, y_use, test_size=0.2, random_state=42)
            scaler = RobustScaler(); X_train_s = scaler.fit_transform(X_train); X_val_s = scaler.transform(X_val)
            # wide
            wide = LinearRegression(); wide.fit(X_train, y_train)

            # candidate models
            models = [
                ('ridge', Ridge(alpha=1.0, random_state=42)),
                ('rf', RandomForestRegressor(n_estimators=500, max_depth=20, n_jobs=-1, random_state=42)),
                ('hgb', HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42))
            ]
            for name, mdl in models:
                try:
                    if name == 'ridge':
                        mdl.fit(X_train_s, y_train)
                        pred_deep_train = mdl.predict(X_train_s)
                        pred_deep_val = mdl.predict(X_val_s)
                    elif name == 'rf':
                        mdl.fit(X_train, y_train)
                        pred_deep_train = mdl.predict(X_train)
                        pred_deep_val = mdl.predict(X_val)
                    else:
                        mdl.fit(X_train, y_train)
                        pred_deep_train = mdl.predict(X_train)
                        pred_deep_val = mdl.predict(X_val)

                    # meta on train preds
                    meta = LinearRegression(); meta.fit(np.vstack([wide.predict(X_train), pred_deep_train]).T, y_train)
                    pred_meta_val = meta.predict(np.vstack([wide.predict(X_val), pred_deep_val]).T)
                    mse, rmse, r2 = metrics(y_val, pred_meta_val)
                    # if logt was used, convert predictions back for metrics on original scale
                    if logt:
                        # compute metrics on original scale by inverse transform
                        # approximate by predicting and then applying expm1
                        pred_meta_val_orig = np.expm1(pred_meta_val)
                        y_val_orig = np.expm1(y_val)
                        mse_o, rmse_o, r2_o = metrics(y_val_orig, pred_meta_val_orig)
                    else:
                        r2_o = r2
                        rmse_o = rmse
                    rec = {'file': fname, 'features_poly': use_poly, 'log_target': logt, 'model': name, 'r2': float(r2_o), 'rmse': float(rmse_o)}
                    tried_records.append(rec)
                    print('  poly=', use_poly, 'logt=', logt, 'model=', name, 'R2=', rec['r2'])
                    # if meets target save model
                    if rec['r2'] >= TARGET_R2:
                        out = {'file': fname, 'features_poly': use_poly, 'log_target': logt, 'model': name, 'r2': rec['r2'], 'rmse': rec['rmse']}
                        joblib.dump({'scaler': scaler, 'wide': wide, 'deep': mdl, 'meta': meta, 'features': list(X.columns)}, MO_DIR / f'fast_model_{Path(fname).stem}.joblib')
                        print('  -> Target reached, saved model for', fname)
                        summary.append(out)
                        # break out fully
                        break
                except Exception as e:
                    print('   model error', name, e)
            else:
                continue
            break
        else:
            continue
        break
    # record best
    if not any(s['file']==fname for s in summary):
        # choose best r2
        if tried_records:
            best = max(tried_records, key=lambda x: x['r2'])
            summary.append(best)

# save summary
(MO_DIR / 'fast_try_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print('\nDone. Summary:', summary)

