# -*- coding: utf-8 -*-
"""
从 data/original_data 中提取每个 Excel 文件中“统计”表的“总通风率”行。
仅保留列：最大值、最小值、平均值、标偏值、变异系数
输出 CSV 到 data/statistic_data，文件名与源文件同名（后缀 .csv），使用 UTF-8-SIG 编码。

使用方式：
    python scripts\extract_statistics.py

说明：脚本会在前 40 行内尝试自动定位表头（包含关键列名），并对列名做宽松匹配。
"""

from pathlib import Path
import sys
import re
import warnings
import numpy as np

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("缺少依赖 pandas：请先运行 `pip install -r requirements.txt` 或安装 pandas") from e

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / 'data' / 'original_data'
OUTPUT_DIR = BASE_DIR / 'data' / 'statistic_data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 要保留的标准列名
WANTED = ['最大值', '最小值', '平均值', '标偏值', '变异系数']
# 用于匹配的指标名称
KEY_METRIC = '总通风率'

# 用于识别表头中各列的关键词映射
COL_KEYWORDS = {
    '最大值': ['最大', '最大值'],
    '最小值': ['最小', '最小值'],
    '平均值': ['平均', '均值', '平均值'],
    '标偏值': ['标偏', '标准偏差', '标偏值'],
    '变异系数': ['变异', '变异系数'],
    '指标': ['指标', '项目', '名称']
}

# 将原始列名映射到标准列名（如果匹配不到则返回原字符串）

def map_column_name(orig):
    if pd.isna(orig):
        return orig
    s = str(orig).strip()
    for std, keywords in COL_KEYWORDS.items():
        for kw in keywords:
            if kw in s:
                return std
    return s


def find_stat_sheet(sheet_names):
    """在工作表名列表中查找第一个包含“统计”或类似字样的表名。"""
    for s in sheet_names:
        if isinstance(s, str) and ('统计' in s or '统计表' in s):
            return s
    # 备用：如果有表名包含 'stat' (英文) 也尝试
    for s in sheet_names:
        if isinstance(s, str) and re.search(r'stat', s, re.I):
            return s
    return None


def detect_header_and_df(excel_path, sheet_name):
    """读取指定 sheet（不设 header），在前 40 行检测包含关键列名的行作为 header。
    返回 pandas.DataFrame（columns 已设置）
    """
    # 先读取无表头形式
    try:
        raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    except Exception:
        # 最后尝试不指定 sheet 名称（让 pandas 自动选择第一个）
        raw = pd.read_excel(excel_path, header=None)

    max_check = min(40, len(raw))
    header_row = None
    header_keywords = sum(COL_KEYWORDS.values(), []) + ['最大', '最小', '平均', '变异', '标偏', '指标']

    for i in range(max_check):
        row = raw.iloc[i].astype(str).tolist()
        joined = '\t'.join(row)
        if any(kw in joined for kw in header_keywords):
            header_row = i
            break

    if header_row is None:
        # 如果未检测到，尝试用 pandas 自动解析（默认第一行为 header）
        try:
            df_auto = pd.read_excel(excel_path, sheet_name=sheet_name)
            return df_auto
        except Exception:
            # 最后退回：把原始数据当作没有表头的数据，直接返回（列为数字索引）
            return raw

    header = raw.iloc[header_row].astype(str).map(lambda x: x.strip())
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = header
    data = data.reset_index(drop=True)
    return data


def standardize_columns(df):
    """把 DataFrame 的列名映射为标准列名，并合并可能重复映射后的列（保留第一个非空）。"""
    orig_cols = list(df.columns)
    mapped = {c: map_column_name(c) for c in orig_cols}
    df = df.rename(columns=mapped)

    # 合并同名列
    cols = list(df.columns)
    for col in cols:
        same = [c for c in cols if c == col]
        if len(same) > 1:
            # 从左到右取第一个非空值
            df[col] = df[same].bfill(axis=1).iloc[:, 0]
            for extra in same[1:]:
                df.drop(columns=extra, inplace=True)
    return df


def process_file(path: Path):
    name = path.name
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        print(f"跳过 {name}：无法读取文件（{e}）")
        return

    sheet = find_stat_sheet(xls.sheet_names)
    if sheet is None:
        print(f"跳过 {name}：未找到包含 '统计' 的工作表，候选表：{xls.sheet_names}")
        return

    try:
        df = detect_header_and_df(path, sheet)
    except Exception as e:
        print(f"读取 {name} 的 {sheet} 表失败：{e}")
        return

    df = standardize_columns(df)

    if '指标' not in df.columns:
        # 如果没有 '指标' 列，尝试把第一列当做指标列
        first_col = df.columns[0]
        warnings.warn(f"{name}: 未找到 '指标' 列，使用第一列 '{first_col}' 作为指标列。")
        df = df.rename(columns={first_col: '指标'})

    # 过滤 指标 == KEY_METRIC
    mask = df['指标'].astype(str).str.strip() == KEY_METRIC
    df_filtered = df[mask]

    if df_filtered.empty:
        print(f"{name}: 未找到 指标 = {KEY_METRIC} 的行，已跳过。")
        return

    # 构造输出表格，仅保留 WANTED 列，若缺失用空值填充
    out = pd.DataFrame()
    for col in WANTED:
        if col in df_filtered.columns:
            out[rename_col(col)] = df_filtered[col].reset_index(drop=True)
        else:
            out[rename_col(col)] = pd.NA
    out['N'] = np.arange(len(out))

    out_name = path.stem + '.csv'
    out_path = OUTPUT_DIR / out_name
    try:
        out.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"已写入: {out_path} (rows={len(out)})")
    except Exception as e:
        print(f"写入 CSV 失败 {out_path}: {e}")
#重命名列
def rename_col(col_name:str):
    if col_name == '最大值':
        return 'Max_Ve'
    if col_name == '最小值':
        return 'Min_Ve'
    if col_name == '平均值':
        return 'Avg_Ve'
    if col_name == '标偏值':
        return 'Std_Ve'
    if col_name == '变异系数':
        return 'is_error'
    return col_name

def main():
    if not INPUT_DIR.exists():
        print(f"输入目录不存在: {INPUT_DIR}")
        sys.exit(1)

    files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in ('.xls', '.xlsx') and p.is_file()])
    if not files:
        print(f"在 {INPUT_DIR} 中未找到 Excel 文件。")
        return

    for p in files:
        process_file(p)


if __name__ == '__main__':
    main()

