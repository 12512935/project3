# -*- coding: utf-8 -*-
"""
针对拟合优度低的模型进行诊断与改进：
- 从 data/model_objects/verification_summary.csv 中找出 val_r2_actual < 0.1 的文件
- 对这些文件用原始训练/验证切分（通过 meta_<stem>.json 中的 val_indices）构建数据集
- 增加简单特征工程：N (若存在)、M*L、M^2、L^2
- 训练并比较：MLP (原有配置)、RandomForestRegressor、GradientBoostingRegressor
- 选择在验证集上表现最好的模型（按 val_r2 优先），保存为 model_improved_<stem>.joblib 并写 meta_improved_<stem>.json
- 输出改进结果汇总到 data/model_objects/improvement_summary.csv
"""
from pathlib import Path
import json
import sys

try:
    import pandas as pd
    import numpy as np
except Exception as e:
    raise RuntimeError('缺少 pandas 或 numpy') from e

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
except Exception as e:
    raise RuntimeError('缺少 scikit-learn 或 joblib') from e

BASE = Path(__file__).resolve().parents[1]
CLEAN_DIR = BASE / 'data' / 'cleaned_data'
MO_DIR = BASE / 'data' / 'model_objects'

VERIF = MO_DIR / 'verification_summary.csv'
THRESH = 0.1  # 如果 val_r2_actual < THRESH 则认为拟合优度低，需要改进

if not VERIF.exists():
    print('未找到 verification_summary.csv，先运行验证脚本。')
    sys.exit(1)

ver = pd.read_csv(VERIF, encoding='utf-8-sig')
# 选择拟合优度低的文件
candidates = ver[ver['val_r2_actual'] < THRESH]['file'].tolist()
if not candidates:
    print('没有检测到 val_r2 <', THRESH, '的模型，任务完成。')
    sys.exit(0)

summary_rows = []
for fname in candidates:
    stem = Path(fname).stem
    print('=== 处理:', fname)
    # 加载 meta 获取 val_indices
    meta_path = MO_DIR / f'meta_{stem}.json'
    if not meta_path.exists():
        print('  未找到 meta 文件，跳过', meta_path)
        continue
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    val_indices = meta.get('val_indices')

    # 读取 cleaned data
    data_path = CLEAN_DIR / fname
    if not data_path.exists():
        print('  未找到 cleaned data 文件，跳过', data_path)
        continue
    df = pd.read_csv(data_path, encoding='utf-8-sig')

    # ensure relevant cols
    required = ['M','C','d','L','R']
    if not set(required).issubset(df.columns):
        print('  数据缺少必要列，跳过')
        continue

    # drop NaN rows (only keep required cols; do NOT use 'N' or 'T' as features)
    df = df[required].dropna()

    # If val_indices from meta may contain indices not present after dropna, intersect
    if val_indices:
        val_idx = [i for i in val_indices if i in df.index]
        if len(val_idx) == 0:
            print('  meta 中的验证索引与数据不匹配，使用尾部样本作为验证集')
            val_count = max(1, int(round(0.2 * len(df))))
            val_idx = list(df.index[-val_count:])
    else:
        print('  meta 中没有 val_indices，使用随机划分')
        val_count = max(1, int(round(0.2 * len(df))))
        val_idx = list(df.index[-val_count:])

    train_idx = [i for i in df.index if i not in val_idx]
    if len(train_idx) < 5 or len(val_idx) < 1:
        print('  训练/验证样本太少，跳过')
        continue

    # 特征工程
    def make_features(ddf):
        X = pd.DataFrame()
        X['M'] = ddf['M'].astype(float)
        X['C'] = ddf['C'].astype(float)
        X['d'] = ddf['d'].astype(float)
        X['L'] = ddf['L'].astype(float)
        # interactions / polynomial
        X['M_L'] = X['M'] * X['L']
        X['M2'] = X['M'] ** 2
        X['L2'] = X['L'] ** 2
        return X

    X_train = make_features(df.loc[train_idx])
    y_train = df.loc[train_idx, 'R'].astype(float).values
    X_val = make_features(df.loc[val_idx])
    y_val = df.loc[val_idx, 'R'].astype(float).values

    # 需要对数值稳定性做检查
    # 尝试三个模型
    models = {}
    # MLP (with scaler)
    mlp = Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(64,32), max_iter=1000, random_state=42))])
    models['MLP'] = mlp
    # RandomForest
    models['RF'] = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    # GradientBoosting
    models['GB'] = GradientBoostingRegressor(n_estimators=200, random_state=42)

    results = []
    for label, model in models.items():
        print(f"  训练模型: {label}")
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            mse = float(mean_squared_error(y_val, pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_val, pred))
            results.append((label, model, mse, rmse, r2))
            print(f"    val_mse={mse:.6f}, val_rmse={rmse:.6f}, val_r2={r2:.6f}")
        except Exception as e:
            print('    训练失败:', e)

    if not results:
        print('  没有成功训练的模型，跳过')
        continue

    # 选择最好模型（按 r2 最大）
    best = max(results, key=lambda x: x[4])
    best_label, best_model, best_mse, best_rmse, best_r2 = best
    print(f"  最佳模型: {best_label} (val_r2={best_r2:.6f})")

    # 保存改进后的模型与 meta
    model_out = MO_DIR / f'model_improved_{stem}.joblib'
    meta_out = MO_DIR / f'meta_improved_{stem}.json'
    try:
        joblib.dump(best_model, model_out)
        meta_new = {
            'file': fname,
            'original_val_r2': float(ver[ver['file']==fname]['val_r2_actual'].iloc[0]),
            'improved_model': best_label,
            'improved_val_mse': best_mse,
            'improved_val_rmse': best_rmse,
            'improved_val_r2': best_r2,
            'features_used': list(X_train.columns),
            'train_size': len(train_idx),
            'val_size': len(val_idx)
        }
        meta_out.write_text(json.dumps(meta_new, ensure_ascii=False, indent=2), encoding='utf-8')
        print('  改进模型与 meta 已保存')
    except Exception as e:
        print('  保存失败:', e)

    summary_rows.append({
        'file': fname,
        'orig_val_r2': float(ver[ver['file']==fname]['val_r2_actual'].iloc[0]),
        'improved_model': best_label,
        'improved_val_r2': best_r2,
        'improved_val_rmse': best_rmse,
        'features': ';'.join(list(X_train.columns))
    })

# 写出汇总
if summary_rows:
    out = pd.DataFrame(summary_rows)
    out_path = MO_DIR / 'improvement_summary.csv'
    out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print('已写入改进汇总:', out_path)
else:
    print('没有改进记录')
