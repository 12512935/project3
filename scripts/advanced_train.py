# -*- coding: utf-8 -*-
"""
Advanced training for a single cleaned_data CSV.
- Adds interaction features and polynomial degree 2
- Optional outlier removal (target bounds)
- Uses a StackingRegressor with HistGradientBoosting, RandomForest and Ridge as base learners
- Evaluates on a held-out validation set and saves model if val R2 >= target

Usage:
  python advanced_train.py --file YXY.csv --target_r2 0.5
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

BASE = Path(__file__).resolve().parents[1]
CLEAN_DIR = BASE / 'data' / 'cleaned_data'
MO_DIR = BASE / 'data' / 'model_objects'
MO_DIR.mkdir(parents=True, exist_ok=True)


def build_features(df, use_poly=True):
    df = df.copy()
    df = df.dropna(subset=['M','C','d','L','R'])
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
    # interactions
    X['M_times_C'] = X['M'] * X['C']
    X['M_times_L'] = X['M'] * X['L']
    X['C_times_d'] = X['C'] * X['d']
    X['L2'] = X['L']**2

    if use_poly:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        Xp = pd.DataFrame(poly.fit_transform(X.values), columns=poly.get_feature_names_out(X.columns), index=X.index)
        return Xp, df['R'].astype(float).values
    return X, df['R'].astype(float).values


def metrics(y_true,y_pred):
    mse = mean_squared_error(y_true,y_pred)
    return mse, float(np.sqrt(mse)), float(r2_score(y_true,y_pred))


def train_and_evaluate(path, target_r2=0.5, random_state=42):
    df = pd.read_csv(path, encoding='utf-8-sig')
    X, y = build_features(df, use_poly=True)

    # remove rows with NaN
    mask = ~np.isnan(X.to_numpy()).any(axis=1)
    X = X.loc[mask]
    y = y[mask]

    # remove outliers in target (keep within 3 std)
    y_mean = y.mean(); y_std = y.std()
    keep_mask = (y >= y_mean - 3*y_std) & (y <= y_mean + 3*y_std)
    X = X.loc[keep_mask]
    y = y[keep_mask]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # base learners
    hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=random_state)
    rf = RandomForestRegressor(n_estimators=300, max_depth=20, n_jobs=-1, random_state=random_state)
    ridge = Ridge(alpha=1.0, random_state=random_state)

    estimators = [('hgb', hgb), ('rf', rf), ('ridge', Pipeline([('scaler', StandardScaler()), ('ridge', ridge)]))]

    stack = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), passthrough=True, n_jobs=-1)

    # fit on scaled for ridge inside pipeline will re-scale; HGB/RF accept both scaled and unscaled, use original X_train and scaled where applicable
    # To keep things simple, fit stack on original features (StackingRegressor will call fit on each estimator accordingly)
    stack.fit(X_train.values, y_train)

    pred_val = stack.predict(X_val.values)
    mse, rmse, r2 = metrics(y_val, pred_val)

    print('Val samples:', len(y_val), 'RMSE=', rmse, 'R2=', r2)

    out = {'file': path.name, 'val_size': int(len(y_val)), 'rmse': rmse, 'r2': r2}

    if r2 >= target_r2:
        # save model including scaler and feature columns
        model_obj = {'model': stack, 'scaler': scaler, 'features': list(X.columns)}
        joblib.dump(model_obj, MO_DIR / f'advanced_model_{Path(path).stem}.joblib')
        out['model_saved'] = True
        print('Saved advanced model for', path.name)
    else:
        out['model_saved'] = False

    (MO_DIR / f'advanced_summary_{Path(path).stem}.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='YXY.csv')
    parser.add_argument('--target_r2', type=float, default=0.5)
    args = parser.parse_args()

    path = CLEAN_DIR / args.file
    if not path.exists():
        print('file not found:', path)
        return
    out = train_and_evaluate(path, target_r2=args.target_r2)
    print('Result:', out)

if __name__ == '__main__':
    main()

