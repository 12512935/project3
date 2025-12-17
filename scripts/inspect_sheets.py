# -*- coding: utf-8 -*-
"""
打印每个 .xls 文件的工作表名，并读取第一个匹配“明细”表的前 8 行（不解析 header），帮助识别列标题和偏移。
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / 'data' / 'original_data'

for path in sorted(INPUT_DIR.glob('*.xls')):
    print('\n===', path.name, '===')
    try:
        xls = pd.ExcelFile(path, engine='xlrd')
        print('sheets:', xls.sheet_names)
        match = None
        for s in xls.sheet_names:
            if '明细' in str(s):
                match = s
                break
        if match is None:
            print('  未找到包含 "明细" 的表，跳过')
            continue
        print('  使用表:', match)
        df = pd.read_excel(path, sheet_name=match, engine='xlrd', header=None)
        print(df.head(8).to_string(index=False, header=False))
    except Exception as e:
        print('  读取失败:', e)

