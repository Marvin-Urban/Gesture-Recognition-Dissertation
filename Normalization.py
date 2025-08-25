import numpy as np
import pandas as pd

def robust_zscore(df: pd.DataFrame) -> pd.DataFrame:
    med = df.median()
    iqr = df.quantile(0.75) - df.quantile(0.25)
    return (df - med) / (iqr.replace(0, 1e-8))

def paper_exact_minmax(df: pd.DataFrame, subject_col: str, sensor_cols: list) -> pd.DataFrame:
    out = []
    for sid, grp in df.groupby(subject_col):
        mins = grp[sensor_cols].min()
        global_max = grp[sensor_cols].max().max()
        denom = (global_max - mins).replace(0, 1.0)
        scaled = (grp[sensor_cols] - mins) / denom
        out.append(pd.concat([scaled, grp.drop(columns=sensor_cols)], axis=1))
    return pd.concat(out, axis=0)

def per_sensor_minmax(df: pd.DataFrame, subject_col: str, sensor_cols: list) -> pd.DataFrame:
    out = []
    for sid, grp in df.groupby(subject_col):
        mins = grp[sensor_cols].min()
        maxs = grp[sensor_cols].max()
        denom = (maxs - mins).replace(0, 1.0)
        scaled = (grp[sensor_cols] - mins) / denom
        out.append(pd.concat([scaled, grp.drop(columns=sensor_cols)], axis=1))
    return pd.concat(out, axis=0)

def l2_normalize(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.to_numpy(dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pd.DataFrame(arr / norms, columns=df.columns)
