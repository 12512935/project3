# -*- coding: utf-8 -*-
"""
Optimize single CSV: feature engineering + RandomizedSearchCV for tree models + meta fusion with wide
Saves best model if val R2 >= target.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform
import joblib

BASE = Path(__file__).resolve().parents[1]
CLEAN_DIR = BASE / 'data' / 'cleaned_data'
MO_DIR = BASE / 'data' / 'model_objects'
MO_DIR.mkdir(parents=True, exist_ok=True)


def build_features(df):
    required = ['M','C','d','L','R']
    df = df[required].dropna().copy()
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
    return mse, float(np.sqrt(mse)), float(r2_score(y_true, y_pred))


def run_search(X_train, y_train, X_val, y_val, random_state=42, n_iter_rf=30, n_iter_hgb=20):
    best = None
    best_score = -999
    tried = []

    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    rf_dist = {'n_estimators': [200, 500, 800], 'max_depth': [None, 10, 20, 40], 'min_samples_split': randint(2, 10)}
    try:
        rs = RandomizedSearchCV(rf, rf_dist, n_iter=min(n_iter_rf, 30), cv=3, random_state=random_state, n_jobs=-1)
        rs.fit(X_train, y_train)
        model = rs.best_estimator_
        pred_meta = LinearRegression().fit(np.vstack([LinearRegression().fit(X_train, y_train).predict(X_train), model.predict(X_train)]).T, y_train).predict(np.vstack([LinearRegression().fit(X_train, y_train).predict(X_val), model.predict(X_val)]).T)
        _, rmse, r2 = metrics(y_val, pred_meta)
        tried.append({'method':'rf','best_params':rs.best_params_, 'r2': r2, 'rmse': rmse})
        if r2 > best_score:
            best_score = r2
            best = ('rf', model, r2, rmse, rs.best_params_)
    except Exception as e:
        tried.append({'method':'rf','error': str(e)})

    hgb = HistGradientBoostingRegressor(random_state=random_state)
    hgb_dist = {'max_iter': [100,200,500], 'learning_rate': [0.01,0.05,0.1], 'max_leaf_nodes':[15,31,63]}
    try:
        rs2 = RandomizedSearchCV(hgb, hgb_dist, n_iter=min(n_iter_hgb,20), cv=3, random_state=random_state, n_jobs=-1)
        rs2.fit(X_train, y_train)
        model2 = rs2.best_estimator_
        pred_meta2 = LinearRegression().fit(np.vstack([LinearRegression().fit(X_train, y_train).predict(X_train), model2.predict(X_train)]).T, y_train).predict(np.vstack([LinearRegression().fit(X_train, y_train).predict(X_val), model2.predict(X_val)]).T)
        _, rmse2, r22 = metrics(y_val, pred_meta2)
        tried.append({'method':'hgb','best_params':rs2.best_params_, 'r2': r22, 'rmse': rmse2})
        if r22 > best_score:
            best_score = r22
            best = ('hgb', model2, r22, rmse2, rs2.best_params_)
    except Exception as e:
        tried.append({'method':'hgb','error': str(e)})

    return best, best_score, tried


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='YXY.csv')
    parser.add_argument('--use_poly', action='store_true')
    parser.add_argument('--log_target', action='store_true')
    parser.add_argument('--remove_outliers', action='store_true')
    parser.add_argument('--n_iter_rf', type=int, default=20)
    parser.add_argument('--n_iter_hgb', type=int, default=10)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--target_r2', type=float, default=0.5)
    args = parser.parse_args()

    path = CLEAN_DIR / args.file
    if not path.exists():
        raise SystemExit('file not found: '+str(path))

    df = pd.read_csv(path, encoding='utf-8-sig')
    X, y, idx = build_features(df)
    print('N samples:', X.shape[0], 'N features:', X.shape[1])

    if args.remove_outliers:
        mu = y.mean(); sigma = y.std()
        mask = (y <= mu + 3*sigma) & (y >= mu - 3*sigma)
        X = X.loc[mask]
        y = y[mask]
        print('After outlier removal, samples:', X.shape[0])

    if args.use_poly:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X = pd.DataFrame(poly.fit_transform(X.values), columns=poly.get_feature_names_out(X.columns), index=X.index)
        print('Poly features count:', X.shape[1])

    if args.log_target:
        y = np.log1p(y)
        print('Applied log1p to target.')

    # split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, shuffle=True)
    print('Train/Val:', X_train.shape[0], X_val.shape[0])

    # scale features
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    best, best_score, tried = run_search(X_train, y_train, X_val, y_val, random_state=args.random_state, n_iter_rf=args.n_iter_rf, n_iter_hgb=args.n_iter_hgb)

    print('Tried:', tried)
    print('Best:', best)

    if best and best_score >= args.target_r2:
        method, model, r2, rmse, params = best
        out = {'file': args.file, 'method': method, 'r2': r2, 'rmse': rmse, 'params': params}
        out_path = MO_DIR / f'opt_best_{Path(args.file).stem}.json'
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
        # save model
        joblib.dump({'scaler': scaler, 'wide': LinearRegression().fit(X_train, y_train), 'deep': model, 'meta': LinearRegression().fit(np.vstack([LinearRegression().fit(X_train, y_train).predict(X_train), model.predict(X_train)]).T, y_train)}, MO_DIR / f'opt_model_{Path(args.file).stem}.joblib')
        print('Saved best model and meta to', MO_DIR)

if __name__ == '__main__':
    main()

