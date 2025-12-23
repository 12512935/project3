# -*- coding: utf-8 -*-
"""
Merge statistic_data into cleaned_data by numeric N match (cleaned N floats -> rounded int match stat N int).

Behavior:
 - read cleaned_data/<name>.csv and statistic_data/<name>.csv
 - detect 'N' columns in both (case-insensitive)
 - convert cleaned N to int by rounding floats, convert stat N to int
 - left-join statistic rows (unique per N) onto cleaned rows by integer N
 - backup original cleaned file to data/cleaned_data_backup/<name> if not already backed up
 - write merged CSV back to data/cleaned_data/<name> (UTF-8 with BOM)
 - record summary to data/model_objects/merge_statistic_summary.csv and JSON

This implements the user's requirement: only match rows where numeric N values are equal.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CLEAN_DIR = BASE / 'data' / 'cleaned_data'
STAT_DIR = BASE / 'data' / 'statistic_data'
MO_DIR = BASE / 'data' / 'model_objects'
BACKUP_DIR = BASE / 'data' / 'cleaned_data_backup'

MO_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

summary = []

csvs = sorted(CLEAN_DIR.glob('*.csv'))
for p in csvs:
    name = p.name
    stat_p = STAT_DIR / name
    if not stat_p.exists():
        print(f"Skip {name}: no matching statistic_data file")
        summary.append({'file': name, 'status': 'no_stat_file'})
        continue

    # prefer original backup if exists
    backup_path = BACKUP_DIR / name
    src_clean_path = backup_path if backup_path.exists() else p

    def read_csv_try(path):
        try:
            return pd.read_csv(path, encoding='utf-8-sig')
        except Exception:
            return pd.read_csv(path, encoding='latin1')

    df_clean = read_csv_try(src_clean_path)
    df_stat = read_csv_try(stat_p)

    # find N column (case-insensitive)
    def find_N_col(df):
        for c in df.columns:
            if str(c).strip().lower() == 'n':
                return c
        return None

    n_clean = find_N_col(df_clean)
    n_stat = find_N_col(df_stat)
    if n_clean is None or n_stat is None:
        print(f"Skip {name}: missing N column in cleaned({n_clean}) or statistic({n_stat})")
        summary.append({'file': name, 'status': 'missing_N'})
        continue

    # work on copies
    dfc = df_clean.copy()
    dfs = df_stat.copy()

    # convert N to integer keys
    def to_int_round(series):
        out = []
        for v in series:
            try:
                f = float(v)
                out.append(int(round(f)))
            except Exception:
                out.append(pd.NA)
        return pd.Series(out, index=series.index)

    dfc['_N_int'] = to_int_round(dfc[n_clean])
    # for statistic data, try to convert directly to int (they are expected to be ints)
    def to_int_direct(series):
        out = []
        for v in series:
            try:
                # if it's float like 1.0 -> int 1
                f = float(v)
                out.append(int(round(f)))
            except Exception:
                out.append(pd.NA)
        return pd.Series(out, index=series.index)

    dfs['_N_int'] = to_int_direct(dfs[n_stat])

    # aggregate statistic data by integer N (take first if duplicates)
    dfs_unique = dfs.groupby('_N_int', as_index=False).first()

    # perform left join on integer N
    merged = pd.merge(dfc, dfs_unique, left_on='_N_int', right_on='_N_int', how='left', suffixes=('','_stat'))

    # determine how many cleaned rows got a stat match
    stat_cols = [c for c in dfs_unique.columns if c != '_N_int']
    stat_cols_after = [c if c in merged.columns else c + '_stat' for c in stat_cols]
    matched = 0
    if stat_cols_after:
        matched = merged[stat_cols_after].notnull().any(axis=1).sum()

    # backup original cleaned file if not already backed up
    if not backup_path.exists():
        try:
            p.rename(backup_path)
        except Exception:
            pass

    # drop helper column(s)
    drop_cols = [c for c in merged.columns if c in ['_N_int']]
    merged = merged.drop(columns=drop_cols)

    # save merged back
    out_path = CLEAN_DIR / name
    merged.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"Merged {name}: cleaned_rows={len(dfc)}, stat_rows={len(dfs)}, matched={int(matched)}")
    summary.append({'file': name, 'status': 'merged', 'cleaned_rows': len(dfc), 'stat_rows': len(dfs), 'matched': int(matched)})

# write summary
summary_df = pd.DataFrame(summary)
summary_df.to_csv(MO_DIR / 'merge_statistic_summary.csv', index=False, encoding='utf-8-sig')
(MO_DIR / 'merge_statistic_summary.json').write_text(summary_df.to_json(orient='records', force_ascii=False), encoding='utf-8')
print('Done. Summary written to data/model_objects/merge_statistic_summary.csv')
