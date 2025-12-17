# -*- coding: utf-8 -*-
"""
为 data/cleaned_data 中的每个 CSV 训练一个前馈神经网络（FNN）回归模型：
- 输入特征：M, C, d, L
- 输出目标：R

实现细节：
- 使用 sklearn pipeline：StandardScaler + MLPRegressor
- 对每个文件执行 train/test 划分（默认 80/20），若样本很少会自动缩小网络或跳过训练
- 将训练好的 model pipeline（包含 scaler）以 joblib 保存为 data/model_objects/model_<name>.joblib
- 将训练元信息（超参数、样本数、训练/测试指标等）保存为 data/model_objects/meta_<name>.json

用法：
    python scripts\model_training.py

依赖：scikit-learn, pandas, numpy, joblib
"""
from pathlib import Path
import json
import sys
import warnings
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
except Exception as e:
    raise RuntimeError("缺少 pandas 或 numpy，请先安装依赖") from e

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
except Exception as e:
    raise RuntimeError("缺少 scikit-learn 或 joblib，请先安装：pip install scikit-learn joblib") from e

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / 'data' / 'cleaned_data'
OUTPUT_DIR = BASE_DIR / 'data' / 'model_objects'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ['M', 'C', 'd', 'L']
TARGET_COL = 'R'


def build_mlp_for_n_samples(n_samples, random_state=42):
    """根据样本量返回合适的 MLPRegressor 配置。"""
    if n_samples < 10:
        return None
    if n_samples < 30:
        hidden = (16,)
    elif n_samples < 100:
        hidden = (32,)
    elif n_samples < 1000:
        hidden = (64, 32)
    else:
        hidden = (128, 64)
    mlp = MLPRegressor(hidden_layer_sizes=hidden,
                       activation='relu',
                       solver='adam',
                       max_iter=1000,
                       random_state=random_state)
    return mlp


def process_file(path: Path):
    name = path.name
    stem = path.stem
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except Exception as e:
        print(f"跳过 {name}: 无法读取 CSV ({e})")
        return

    # 检查列
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        print(f"{name}: 缺少列 {missing}，已跳过。")
        return

    # 丢弃包含缺失的行
    df = df[FEATURE_COLS + [TARGET_COL]].dropna()
    n_samples = len(df)

    if n_samples == 0:
        print(f"{name}: 无有效样本，跳过。")
        return

    mlp = build_mlp_for_n_samples(n_samples)
    meta = {
        'file': name,
        'n_samples': n_samples,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'skipped': False,
    }

    if mlp is None:
        meta.update({'skipped': True, 'reason': '样本数不足 (<10)'} )
        out_meta = OUTPUT_DIR / f"meta_{stem}.json"
        with out_meta.open('w', encoding='utf-8') as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        print(f"{name}: 样本数 {n_samples} < 10，已跳过训练，记录元信息到 {out_meta}")
        return

    X = df[FEATURE_COLS].astype(float).values
    y = df[TARGET_COL].astype(float).values

    # 划分数据：显式保留约 20% 用作验证集，并记录被保留的数据（使用原始 DataFrame 的索引）
    # 计算验证集大小（至少保留 1 个样本）
    val_count = max(1, int(round(0.2 * n_samples)))
    indices = df.index.to_numpy()
    # 使用 train_test_split 切分索引以保持原始索引信息
    train_idx, val_idx = train_test_split(indices, test_size=val_count, random_state=42)

    X_train = df.loc[train_idx, FEATURE_COLS].astype(float).values
    y_train = df.loc[train_idx, TARGET_COL].astype(float).values
    X_val = df.loc[val_idx, FEATURE_COLS].astype(float).values
    y_val = df.loc[val_idx, TARGET_COL].astype(float).values

    # 将验证集的原始行写出，便于审查
    val_out_path = OUTPUT_DIR / f"val_{stem}.csv"
    try:
        df.loc[val_idx, FEATURE_COLS + [TARGET_COL]].to_csv(val_out_path, index=False, encoding='utf-8-sig')
    except Exception as e:
        warnings.warn(f"{name}: 无法保存验证集到 {val_out_path}：{e}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', mlp)
    ])

    # 训练
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        warnings.warn(f"{name}: 训练失败：{e}")
        meta.update({'skipped': True, 'reason': f'train_failed: {e}'})
        out_meta = OUTPUT_DIR / f"meta_{stem}.json"
        with out_meta.open('w', encoding='utf-8') as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        return

    # 评估
    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)
    train_mse = float(mean_squared_error(y_train, y_train_pred))
    val_mse = float(mean_squared_error(y_val, y_val_pred))
    train_r2 = float(r2_score(y_train, y_train_pred))
    val_r2 = float(r2_score(y_val, y_val_pred))

    meta.update({
         'model_params': {
             'hidden_layer_sizes': pipeline.named_steps['mlp'].hidden_layer_sizes,
             'activation': pipeline.named_steps['mlp'].activation,
             'solver': pipeline.named_steps['mlp'].solver,
             'max_iter': pipeline.named_steps['mlp'].max_iter,
         },
        'train_size': len(X_train),
        'val_size': len(X_val),
        'val_indices': [int(x) for x in list(val_idx)],
        'val_file': str(val_out_path.name),
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_r2': train_r2,
        'val_r2': val_r2,
     })

    # 保存 model pipeline
    model_path = OUTPUT_DIR / f"model_{stem}.joblib"
    meta_path = OUTPUT_DIR / f"meta_{stem}.json"
    try:
        joblib.dump(pipeline, model_path)
        with meta_path.open('w', encoding='utf-8') as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        print(f"{name}: 训练完成。模型已保存到 {model_path}，元信息已保存到 {meta_path}")
    except Exception as e:
        warnings.warn(f"{name}: 保存模型或元信息失败：{e}")


def main():
    if not INPUT_DIR.exists():
        print(f"输入目录不存在: {INPUT_DIR}")
        sys.exit(1)

    csv_files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() == '.csv' and p.is_file()])
    if not csv_files:
        print(f"在 {INPUT_DIR} 中未找到 CSV 文件。")
        return

    for p in csv_files:
        process_file(p)


if __name__ == '__main__':
    main()
