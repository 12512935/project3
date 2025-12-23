# -*- coding: utf-8 -*-
"""
Quick try: train wide + tree deep + meta on one cleaned_data CSV
Usage:
    python quick_try.py --file YXY.csv --n_estimators 500 --use_hgb --use_poly --random_state 42
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

BASE = Path(__file__).resolve().parents[1]
CLEAN_DIR = BASE / 'data' / 'cleaned_data'
MO_DIR = BASE / 'data' / 'model_objects'
MO_DIR.mkdir(parents=True, exist_ok=True)


def build_features(df):
    required = ['M', 'C', 'd', 'L', 'R']
    df = df.copy()
    df = df[required].dropna()
    X = pd.DataFrame()
    X['M'] = df['M'].astype(float)
    X['C'] = df['C'].astype(float)
    X['d'] = df['d'].astype(float)
    X['L'] = df['L'].astype(float)
    L_nonzero = df['L'].replace(0, np.nan)
    X['M_over_L'] = (df['M'] / L_nonzero).fillna(0.0).astype(float)
    X['C2'] = (df['C'].astype(float) ** 2)
    y = df['R'].astype(float).values
    return X, y, df.index


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return mse, rmse, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--n_estimators', type=int, default=500)
    parser.add_argument('--use_hgb', action='store_true')
    parser.add_argument('--use_poly', action='store_true')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--target_r2', type=float, default=0.5)
    args = parser.parse_args()

    path = CLEAN_DIR / args.file
    if not path.exists():
        raise SystemExit('file not found: ' + str(path))

    df = pd.read_csv(path, encoding='utf-8-sig')
    X, y, idx = build_features(df)
    if args.use_poly:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        Xp = pd.DataFrame(poly.fit_transform(X.values), columns=poly.get_feature_names_out(X.columns), index=X.index)
        print('Using poly features, count=', Xp.shape[1])
        X = Xp

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, shuffle=True)

    wide = LinearRegression(); wide.fit(X_train, y_train)

    results = []

    # RandomForest
    print('Training RandomForest n=', args.n_estimators)
    rf = RandomForestRegressor(n_estimators=args.n_estimators, n_jobs=-1, random_state=args.random_state)
    rf.fit(X_train, y_train)
    pred_rf_train = rf.predict(X_train)
    pred_rf_val = rf.predict(X_val)
    # meta
    meta = LinearRegression(); meta.fit(np.vstack([wide.predict(X_train), pred_rf_train]).T, y_train)
    pred_meta_val = meta.predict(np.vstack([wide.predict(X_val), pred_rf_val]).T)
    rf_mse, rf_rmse, rf_r2 = metrics(y_val, pred_meta_val)
    print('RF meta: RMSE=%.6f R2=%.6f' % (rf_rmse, rf_r2))
    results.append(('rf', rf_r2, rf_rmse))

    # HGB
    if args.use_hgb:
        print('Training HistGradientBoosting')
        hgb = HistGradientBoostingRegressor(max_iter=200, random_state=args.random_state)
        hgb.fit(X_train, y_train)
        pred_hgb_train = hgb.predict(X_train)
        pred_hgb_val = hgb.predict(X_val)
        meta2 = LinearRegression(); meta2.fit(np.vstack([wide.predict(X_train), pred_hgb_train]).T, y_train)
        pred_meta_val2 = meta2.predict(np.vstack([wide.predict(X_val), pred_hgb_val]).T)
        hgb_mse, hgb_rmse, hgb_r2 = metrics(y_val, pred_meta_val2)
        print('HGB meta: RMSE=%.6f R2=%.6f' % (hgb_rmse, hgb_r2))
        results.append(('hgb', hgb_r2, hgb_rmse))

    best = max(results, key=lambda x: x[1])
    print('Best result:', best)

    # save if meets target
    if best[1] >= args.target_r2:
        out = {'file': args.file, 'best': best}
        out_path = MO_DIR / f'quick_best_{Path(args.file).stem}.json'
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Saved quick_best to', out_path)

if __name__ == '__main__':
    main()

