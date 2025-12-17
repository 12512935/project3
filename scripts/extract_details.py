# -*- coding: utf-8 -*-
"""
从 data/original_data 中读取每个 .xls 文件，提取工作表名为 "明细数据表"（或包含"明细"的工作表）
并仅保留列：质量，圆周，圆度，长度，吸阻，记录时刻，输出到 data/processed_data 下的同名 CSV 文件。

用法：
    python scripts\\extract_details.py

依赖：pandas, xlrd

"""
import os
import sys
from pathlib import Path
from copy import deepcopy
import numpy as np

try:
    import pandas as pd
except Exception as e:
    print("错误：无法导入 pandas。请先运行: pip install -r requirements.txt")
    raise

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / 'data' / 'original_data'
OUTPUT_DIR = BASE_DIR / 'data' / 'processed_data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WANTED_COLUMNS = ['质量', '圆周', '圆度', '长度', '吸阻', '记录时刻']


def _map_column_name(col_name: str) -> str:
    """根据原始列名将其映射为标准列名（中文）。"""
    if not isinstance(col_name, str):
        col_name = str(col_name)
    s = col_name.strip()
    # 使用包含匹配来覆盖多种括号/单位写法
    if '重' in s or '质量' in s or '重量' in s:
        return '质量'
    if '圆周' in s:
        return '圆周'
    if '圆度' in s:
        return '圆度'
    if '长度' in s or ('长' in s and '长度' in s):
        return '长度'
    if '吸阻' in s:
        return '吸阻'
    # 记录时刻 / 检测时刻 / 检测时间 归类为记录时刻
    if '记录时刻' in s:
        return '记录时刻'
    return s

def col_rename(col_name: str) -> str:
    if col_name == '质量':
        return 'M'
    if col_name == '圆周':
        return 'C'
    if col_name == '圆度':
        return 'd'
    if col_name == '长度':
        return 'L'
    if col_name == '吸阻':
        return 'R'
    if col_name == '记录时刻':
        return 'T'
    return col_name

def process_file(path: Path):
    """处理单个 Excel 文件并写出 CSV。会智能检测表头所在行并映射列名。"""
    name = path.name
    try:
        xls = pd.ExcelFile(path, engine='xlrd')
        sheets = xls.sheet_names
        match = None
        for s in sheets:
            if '明细' in str(s):
                match = s
                break
        if match is None:
            print(f"跳过 {name}: 未找到包含 '明细' 的工作表；可用工作表: {sheets}")
            return

        df_raw = pd.read_excel(path, sheet_name=match, engine='xlrd', header=None)
    except Exception as e:
        print(f"错误读取 {name}: {e}")
        return

    # 在前 30 行中查找包含关键列名的一行，作为表头行
    header_row = None
    for i in range(min(30, len(df_raw))):
        row_vals = df_raw.iloc[i].astype(str).tolist()
        joined = '\t'.join(row_vals)
        if ('圆周' in joined) or ('圆度' in joined) or ('重量' in joined) or ('序号' in joined):
            header_row = i
            break

    if header_row is None:
        # 退回到 pandas 默认，尝试作为常规表格读取
        try:
            df = pd.read_excel(path, sheet_name=match, engine='xlrd')
        except Exception as e:
            print(f"无法解析 {name} 的表头: {e}")
            return
    else:
        header = df_raw.iloc[header_row].astype(str).map(lambda x: x.strip())
        data = df_raw.iloc[header_row+1:].copy()
        data.columns = header
        df = data.reset_index(drop=True)

    # 规范化并映射列名
    col_map = {}
    for col in df.columns:
        mapped = _map_column_name(str(col))
        col_map[col] = mapped

    df = df.rename(columns=col_map)

    # 合并重复映射到相同名称的列（例如 '检测时刻' 和 '记录时刻' 都映射为 '测量时间'）
    # 对这些重复列取逐行第一个非空值
    cols = list(df.columns)
    consolidated = df.copy()
    seen = set()
    for c in cols:
        if c in seen:
            continue
        duplicates = [col for col in cols if col == c]
        if len(duplicates) > 1:
            # bfill 横向填充并取第一列
            consolidated[c] = consolidated[duplicates].bfill(axis=1).iloc[:, 0]
            # drop other duplicate columns (keep only one)
            for d in duplicates[1:]:
                consolidated.drop(columns=d, inplace=True)
        seen.add(c)
    df = consolidated

    # 构造输出 DataFrame，仅保留需要的列，缺失列用空值填充
    out_df = pd.DataFrame()
    for col in WANTED_COLUMNS:
        if col in df.columns:
            out_df[col_rename(col)] = df[col]
        else:
            out_df[col_rename(col)] = pd.NA
    # 处理记录时刻，添加 N 列
    out_df = process_details(out_df)

    out_name = path.stem + '.csv'
    out_path = OUTPUT_DIR / out_name
    try:
        out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"已写入: {out_path} (rows={len(out_df)})")
    except Exception as e:
        print(f"写入 CSV 失败 {out_path}: {e}")

def process_details(df):
    t = df['T'][0]
    i = 0
    output = deepcopy(df)
    output['N'] = np.zeros(len(output))
    for row in output.iterrows():
        if row[1]['T'] != t:
            i += 1
            t = row[1]['T']
            output.at[row[0],'N'] = i
        else:
            output.at[row[0],'N'] = i
    return output


def main():
    if not INPUT_DIR.exists():
        print(f"输入目录不存在: {INPUT_DIR}")
        sys.exit(1)

    files = sorted(INPUT_DIR.iterdir())
    xls_files = [p for p in files if p.is_file() and p.suffix.lower() == '.xls']
    if not xls_files:
        print(f"在 {INPUT_DIR} 中未找到 .xls 文件。")
        return

    for p in xls_files:
        process_file(p)


if __name__ == '__main__':
    main()
