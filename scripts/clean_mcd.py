# -*- coding: utf-8 -*-
"""
对 data/processed_data 中每个 CSV 文件：
- 读取列 M, C, d, L, R
- 使用 RobustScaler 进行归一化（对中位数和 IQR 标准化，减少异常值影响）
- 使用 Minimum Covariance Determinant (MCD) 计算 Mahalanobis 距离，基于卡方分布阈值或经验分位数筛选出内点
- 将仅包含内点的原始行写到 data/cleaned_data 下同名 CSV

用法:
    python scripts\clean_mcd.py

依赖: pandas, numpy, scikit-learn, (可选) scipy
"""
from pathlib import Path
import sys
import warnings

try:
    import pandas as pd
    import numpy as np
except Exception as e:
    raise RuntimeError("缺少 pandas 或 numpy，请先安装依赖 (pip install pandas numpy)") from e

try:
    from sklearn.covariance import MinCovDet
    from sklearn.preprocessing import RobustScaler
except Exception as e:
    raise RuntimeError("缺少 scikit-learn，请先安装：pip install scikit-learn") from e

# 可选：用于阈值计算
try:
    from scipy.stats import chi2
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# 如果没有 scipy，给出警告（使用 warnings 来避免未使用导入的 lint 警告）
if not _HAS_SCIPY:
    warnings.warn("scipy 未安装：将使用经验分位数作为阈值；要使用卡方分布阈值请安装 scipy（pip install scipy）")

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / 'data' / 'processed_data'
OUTPUT_DIR = BASE_DIR / 'data' / 'cleaned_data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ['M', 'C', 'd', 'L', 'R']


def process_file(path: Path, alpha=0.975):
    name = path.name
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except Exception as e:
        print(f"跳过 {name}：无法读取 CSV（{e}）")
        return

    # 检查特征列
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"{name}: 缺少列 {missing}，已跳过。")
        return

    # 保留原始索引以便写回原始行
    df_orig = df.copy()

    # 仅对没有缺失值的行进行 MCD 计算
    valid_mask = df[FEATURE_COLS].notna().all(axis=1)
    valid_idx = df.index[valid_mask]
    n_valid = len(valid_idx)
    p = len(FEATURE_COLS)

    if n_valid == 0:
        print(f"{name}: 无有效样本，跳过。")
        return

    if n_valid <= p:
        # MCD 无法在样本数 <= 维度时正常工作；直接保留所有有效样本
        print(f"{name}: 有效样本 ({n_valid}) <= 维度 ({p})，无法运行 MCD，保留有效样本并写出。")
        out_df = df_orig.loc[valid_idx]
        out_path = OUTPUT_DIR / name
        out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"已写入: {out_path} (rows={len(out_df)})")
        return

    X = df.loc[valid_idx, FEATURE_COLS].astype(float).values

    # 归一化（使用 RobustScaler 减少异常值对尺度的影响）
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # 计算 MCD
    try:
        mcd = MinCovDet().fit(X_scaled)
    except Exception as e:
        print(f"{name}: MCD 训练失败：{e}，已跳过。")
        return

    # 计算马氏距离的平方
    try:
        # 直接使用 MCD 返回的 location_ 和 covariance_ 计算马氏距离，避免某些静态分析器的警告
        cov = mcd.covariance_
        # 可能需要逆矩阵
        try:
            icov = np.linalg.inv(cov)
        except Exception:
            # 如果协方差矩阵不可逆，使用伪逆
            icov = np.linalg.pinv(cov)
        diff = X_scaled - mcd.location_
        # 广播计算：md2 = diag(diff @ icov @ diff.T)
        md2 = np.einsum('ij,jk,ik->i', diff, icov, diff)
    except Exception as e:
        print(f"{name}: 计算马氏距离失败：{e}，已跳过。")
        return

    # 根据卡方分布确定阈值（如果 scipy 可用），否则用经验分位数
    if _HAS_SCIPY:
        thresh = chi2.ppf(alpha, df=p)
    else:
        # 使用 alpha 分位数（例如 97.5%）的经验阈值
        thresh = np.percentile(md2, alpha * 100)

    inlier_mask = md2 <= thresh
    inlier_idx = valid_idx[inlier_mask]

    # 构造输出 DataFrame：保留原始所有列，但只包含 inlier 行
    out_df = df_orig.loc[inlier_idx].reset_index(drop=True)

    out_path = OUTPUT_DIR / name
    try:
        out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"{name}: 原始行数={len(df_orig)}, 有效样本={n_valid}, 保留行数={len(out_df)} -> {out_path}")
    except Exception as e:
        print(f"写入 {out_path} 失败：{e}")


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
