# -*- coding: utf-8 -*-
"""
训练宽-深 (wide & deep) 模型：
- 特征: M, C, d, L, M/L, C**2
- 输出: R
- Deep 部分使用 MLPRegressor，默认三层 (128,64,32)，可通过命令行参数调整
- Wide: 线性回归
- Meta: 线性回归融合 wide 与 deep 的预测

对 data/cleaned_data 中每个 CSV 文件执行：
- 读取数据（仅使用 M,C,d,L,R 列，严格不使用 N 与 T）
- 划分训练/验证（若存在 meta_<stem>.json 中的 val_indices 则使用它，否则随机 20% 作为验证集）
- 训练模型并评估（MSE, RMSE, R2），保存模型 joblib 与 meta json 到 data/model_objects
- 输出汇总 CSV: data/model_objects/model_wide_deep_summary.csv

用法示例：
    python wide_deep_train.py
可选参数：
    --hidden 128 64 32
    --test_size 0.2
    --random_state 42

"""
from pathlib import Path
import json
import argparse
import sys

try:
    import pandas as pd
    import numpy as np
except Exception as e:
    raise RuntimeError('缺少 pandas 或 numpy') from e

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.linear_model import Ridge
    from scipy.stats import randint, uniform
except Exception as e:
    raise RuntimeError('缺少 scikit-learn 或 joblib 或 scipy') from e

BASE = Path(__file__).resolve().parents[1]
CLEAN_DIR = BASE / 'data' / 'cleaned_data'
MO_DIR = BASE / 'data' / 'model_objects'
MO_DIR.mkdir(parents=True, exist_ok=True)


def build_features(df):
    """从 df 中构造特征矩阵 X（DataFrame）和目标 y（ndarray）。
    只使用 M,C,d,L,R，禁止使用 N 与 T。增加 Max_Ve, Min_Ve, Avg_Ve（如果存在）作为额外输入特征。
    """
    required = ['M', 'C', 'd', 'L', 'R']
    if not set(required).issubset(df.columns):
        raise ValueError(f"输入数据必须包含列: {required}")
    dd = df.copy()
    # drop rows with NaN in required
    dd = dd.dropna(subset=required)
    X = pd.DataFrame()
    X['M'] = dd['M'].astype(float)
    X['C'] = dd['C'].astype(float)
    X['d'] = dd['d'].astype(float)
    X['L'] = dd['L'].astype(float)
    # add statistic features if present, but do not use is_error, N, T
    for extra in ['Max_Ve', 'Min_Ve', 'Avg_Ve']:
        if extra in dd.columns:
            X[extra] = pd.to_numeric(dd[extra], errors='coerce').fillna(0.0)
    # M/L: 避免除以 0
    L_nonzero = dd['L'].replace(0, np.nan)
    X['M_over_L'] = (dd['M'] / L_nonzero).fillna(0.0).astype(float)
    X['C2'] = (dd['C'].astype(float) ** 2)
    y = dd['R'].astype(float).values
    return X, y, dd.index


def train_one(csv_path, hidden_layers=(128, 64, 32), test_size=0.2, random_state=42, max_iter=1000, candidates=None, target_r2=0.5, use_poly=False):
    """对单个 CSV 文件训练宽-深模型并保存结果，返回 summary dict.
    支持传入多个 candidates（list of tuple），按顺序尝试，直到 meta R2 >= target_r2 或尝试完所有。
    返回时保存最佳模型与尝试记录。
    """
    fname = csv_path.name
    stem = csv_path.stem
    print(f"\nProcessing: {fname}")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception:
        df = pd.read_csv(csv_path, encoding='latin1')

    try:
        X, y, indices = build_features(df)
    except Exception as e:
        print('  跳过: 构造特征失败:', e)
        return None

    # check meta for val_indices
    meta_path = MO_DIR / f'meta_{stem}.json'
    val_idx = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            val_indices = meta.get('val_indices')
            if val_indices:
                # intersect with current indices
                val_idx = [i for i in val_indices if i in indices]
                if len(val_idx) == 0:
                    val_idx = None
        except Exception:
            val_idx = None

    if val_idx is not None:
        # use provided val indices
        train_idx = [i for i in indices if i not in val_idx]
        if len(train_idx) < 5 or len(val_idx) < 1:
            print('  meta 中的索引导致训练/验证样本不足，改为随机划分')
            val_idx = None

    if val_idx is None:
        # random split
        X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X, y, indices, test_size=test_size, random_state=random_state, shuffle=True)
        val_idx = list(idx_val)
        train_idx = list(idx_train)
    else:
        train_idx = [i for i in indices if i not in val_idx]
        X_train = X.loc[train_idx]
        X_val = X.loc[val_idx]
        y_train = y[[list(indices).index(i) for i in train_idx]]
        y_val = y[[list(indices).index(i) for i in val_idx]]

    if len(train_idx) < 5 or len(val_idx) < 1:
        print('  训练/验证样本太少，跳过')
        return None

    # prepare feature sets: original and optionally polynomial
    feature_sets = [(X, 'orig')]
    if use_poly:
        try:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly_arr = poly.fit_transform(X.values)
            poly_names = poly.get_feature_names_out(X.columns)
            X_poly = pd.DataFrame(X_poly_arr, columns=poly_names, index=X.index)
            feature_sets.append((X_poly, 'poly2'))
            print('  使用多项式二阶特征, 新特征数:', X_poly.shape[1])
        except Exception as e:
            print('  无法构造多项式特征:', e)

    # candidates handling
    tried = []
    best = None
    best_score = -999
    best_details = None

    if candidates is None:
        candidates = [tuple(hidden_layers)]

    for X_use, tag in feature_sets:
        print(f"  使用特征集: {tag}")
        # split according to indices already made
        X_train_use = X_use.loc[train_idx]
        X_val_use = X_use.loc[val_idx]

        # re-fit wide on chosen features
        wide = LinearRegression()
        wide.fit(X_train_use, y_train)

        for cand in candidates:
            try:
                print(f"    尝试配置: {cand}")
                # Deep model (MLP) with scaler and early stopping
                deep = Pipeline([
                    ('scaler', StandardScaler()),
                    ('mlp', MLPRegressor(hidden_layer_sizes=tuple(cand), max_iter=max_iter,
                                         random_state=random_state, early_stopping=True,
                                         n_iter_no_change=30, validation_fraction=0.1, tol=1e-4))
                ])
                deep.fit(X_train_use, y_train)

                # Meta learner
                pred_wide_train = wide.predict(X_train_use).reshape(-1, 1)
                pred_deep_train = deep.predict(X_train_use).reshape(-1, 1)
                meta_X_train = np.hstack([pred_wide_train, pred_deep_train])
                meta = LinearRegression()
                meta.fit(meta_X_train, y_train)

                # Validation predictions
                pred_wide_val = wide.predict(X_val_use)
                pred_deep_val = deep.predict(X_val_use)
                meta_X_val = np.vstack([pred_wide_val, pred_deep_val]).T
                pred_meta_val = meta.predict(meta_X_val)

                wide_mse, wide_rmse, wide_r2 = mean_squared_error(y_val, pred_wide_val), float(np.sqrt(mean_squared_error(y_val, pred_wide_val))), float(r2_score(y_val, pred_wide_val))
                deep_mse, deep_rmse, deep_r2 = mean_squared_error(y_val, pred_deep_val), float(np.sqrt(mean_squared_error(y_val, pred_deep_val))), float(r2_score(y_val, pred_deep_val))
                meta_mse, meta_rmse, meta_r2 = mean_squared_error(y_val, pred_meta_val), float(np.sqrt(mean_squared_error(y_val, pred_meta_val))), float(r2_score(y_val, pred_meta_val))

                tried.append({
                    'features': tag,
                    'hidden': tuple(cand),
                    'wide': {'mse': wide_mse, 'rmse': wide_rmse, 'r2': wide_r2},
                    'deep': {'mse': deep_mse, 'rmse': deep_rmse, 'r2': deep_r2},
                    'meta': {'mse': meta_mse, 'rmse': meta_rmse, 'r2': meta_r2}
                })

                print(f"      Meta R2={meta_r2:.4f}")

                if meta_r2 > best_score:
                    best_score = meta_r2
                    best = {'wide': wide, 'deep': deep, 'meta': meta, 'feature_columns': list(X_use.columns), 'feature_tag': tag}
                    best_details = tried[-1]

                if meta_r2 >= target_r2:
                    print(f"      达到目标 R2={meta_r2:.4f}，停止尝试更多配置/特征集。")
                    break

            except Exception as e:
                print('      训练失败 for config', cand, e)
                tried.append({'features': tag, 'hidden': tuple(cand), 'error': str(e)})
                continue

        if best_score >= target_r2:
            break

    # If MLP candidates didn't reach target, try tree-based models (with light randomized search)
    if best_score < target_r2:
        print('  MLP + 多项式未达到目标，尝试树模型候选（含随机搜索）...')
        # simple parameter distributions for RF and HGB
        rf_dist = {'n_estimators': [200, 500], 'max_depth': [None, 10, 20], 'min_samples_split': randint(2, 10)}
        hgb_dist = {'max_iter': [200, 500], 'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [31, 63, None]}

        for X_use, tag in feature_sets:
            X_train_use = X_use.loc[train_idx]
            X_val_use = X_use.loc[val_idx]

            # Randomized search RF
            try:
                print('    RandomizedSearchCV for RF on', tag)
                rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
                rs = RandomizedSearchCV(rf, rf_dist, n_iter=12, cv=3, random_state=random_state, n_jobs=-1)
                rs.fit(X_train_use, y_train)
                model = rs.best_estimator_
                print('      RF best params:', rs.best_params_)

                # meta with wide
                pred_wide_train = wide.predict(X_train_use).reshape(-1, 1)
                pred_deep_train = model.predict(X_train_use).reshape(-1, 1)
                meta_X_train = np.hstack([pred_wide_train, pred_deep_train])
                meta = LinearRegression(); meta.fit(meta_X_train, y_train)
                pred_wide_val = wide.predict(X_val_use)
                pred_deep_val = model.predict(X_val_use)
                pred_meta_val = meta.predict(np.vstack([pred_wide_val, pred_deep_val]).T)
                meta_mse, meta_rmse, meta_r2 = mean_squared_error(y_val, pred_meta_val), float(np.sqrt(mean_squared_error(y_val, pred_meta_val))), float(r2_score(y_val, pred_meta_val))
                tried.append({'tree_search': 'rf', 'features': tag, 'best_params': rs.best_params_, 'meta': {'r2': meta_r2}})
                print('      RF meta R2=', meta_r2)
                if meta_r2 > best_score:
                    best_score = meta_r2
                    best = {'wide': wide, 'deep': model, 'meta': meta, 'feature_columns': list(X_use.columns), 'feature_tag': tag}
                    best_details = tried[-1]
                if meta_r2 >= target_r2:
                    break
            except Exception as e:
                print('      RF search failed', e)

            # HistGradientBoosting randomized try
            try:
                print('    RandomizedSearchCV for HGB on', tag)
                hgb = HistGradientBoostingRegressor(random_state=random_state)
                rs2 = RandomizedSearchCV(hgb, hgb_dist, n_iter=8, cv=3, random_state=random_state, n_jobs=-1)
                rs2.fit(X_train_use, y_train)
                model = rs2.best_estimator_
                print('      HGB best params:', rs2.best_params_)

                pred_wide_train = wide.predict(X_train_use).reshape(-1, 1)
                pred_deep_train = model.predict(X_train_use).reshape(-1, 1)
                meta_X_train = np.hstack([pred_wide_train, pred_deep_train])
                meta = LinearRegression(); meta.fit(meta_X_train, y_train)
                pred_wide_val = wide.predict(X_val_use)
                pred_deep_val = model.predict(X_val_use)
                pred_meta_val = meta.predict(np.vstack([pred_wide_val, pred_deep_val]).T)
                meta_mse, meta_rmse, meta_r2 = mean_squared_error(y_val, pred_meta_val), float(np.sqrt(mean_squared_error(y_val, pred_meta_val))), float(r2_score(y_val, pred_meta_val))
                tried.append({'tree_search': 'hgb', 'features': tag, 'best_params': rs2.best_params_, 'meta': {'r2': meta_r2}})
                print('      HGB meta R2=', meta_r2)
                if meta_r2 > best_score:
                    best_score = meta_r2
                    best = {'wide': wide, 'deep': model, 'meta': meta, 'feature_columns': list(X_use.columns), 'feature_tag': tag}
                    best_details = tried[-1]
                if meta_r2 >= target_r2:
                    break
            except Exception as e:
                print('      HGB search failed', e)

        # end feature_sets loop for trees

    if best is None:
        print('  未能训练出有效模型，跳过保存')
        return None

    # Save best model dict
    model_out = MO_DIR / f'model_wide_deep_{stem}.joblib'
    joblib.dump(best, model_out)

    # Save meta
    meta_out = {
        'file': fname,
        'model_file': model_out.name,
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'val_indices': val_idx,
        'best_hidden': best_details.get('hidden') if best_details else None,
        'tried': tried,
        'metrics': best_details
    }
    meta_path_out = MO_DIR / f'meta_wide_deep_{stem}.json'
    meta_path_out.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding='utf-8')

    summary = {
        'file': fname,
        'model_file': model_out.name,
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'val_r2_meta': best_score,
        'val_rmse_meta': best_details.get('meta', {}).get('rmse') if best_details else None,
        'features': ';'.join(list(X.columns)),
        'best_hidden': best_details.get('hidden') if best_details else None
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', nargs='+', type=int, default=(256, 128, 64, 32), help='MLP 神经元数，支持任意层，例如 --hidden 256 128 64 32')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--file', type=str, default=None, help='仅处理指定的 cleaned_data 文件名，例如 HLS.csv')
    parser.add_argument('--limit', type=int, default=None, help='仅处理前 N 个文件（按排序）')
    parser.add_argument('--target_r2', type=float, default=0.5, help='目标验证集 R2，达到后停止在该文件上尝试更多配置')
    parser.add_argument('--poly', action='store_true', help='是否使用多项式二阶特征（增加非线性特征）')
    args = parser.parse_args()

    csvs = sorted(CLEAN_DIR.glob('*.csv'))
    # 如果指定了单个文件名，只处理该文件
    if args.file:
        target = CLEAN_DIR / args.file
        if not target.exists():
            print('指定的文件不存在:', target)
            sys.exit(1)
        csvs = [target]
    # 如果指定了 limit，截取前 N 个
    if args.limit is not None:
        csvs = csvs[:args.limit]

    if not csvs:
        print('未找到 cleaned_data 下的 csv 文件，请先生成清洗后的数据。')
        sys.exit(1)

    # ensure hidden is a tuple
    hidden = tuple(args.hidden) if isinstance(args.hidden, (list, tuple)) else (args.hidden,)

    # build candidate configs: primary is user-provided, then a few larger/deeper variants
    candidates = [hidden]
    # add some common larger/deeper variants if not duplicates
    extras = [ (512,256,128,64), (128,128,64,32), (512,512,256,128), (1024,512,256,128) ]
    for e in extras:
        if e not in candidates:
            candidates.append(e)

    summaries = []
    for p in csvs:
        try:
            s = train_one(p, hidden_layers=hidden, test_size=args.test_size, random_state=args.random_state, max_iter=args.max_iter, candidates=candidates, target_r2=args.target_r2, use_poly=args.poly)
            if s:
                summaries.append(s)
        except Exception as e:
            print('处理失败:', p.name, e)

    if summaries:
        out_df = pd.DataFrame(summaries)
        out_path = MO_DIR / 'model_wide_deep_summary.csv'
        out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print('\nSummary written to', out_path)
    else:
        print('\nNo models trained.')


if __name__ == '__main__':
    main()
