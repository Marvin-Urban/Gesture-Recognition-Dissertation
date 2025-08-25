import numpy as np
import pandas as pd

def adjacent_difference(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.to_numpy(dtype=np.float32)
    diff = arr[:, 1:] - arr[:, :-1]
    cols = [f"adjdiff_{i}" for i in range(diff.shape[1])]
    return pd.DataFrame(diff, columns=cols)

def temporal_slope(df: pd.DataFrame, alpha: float = 0.3) -> pd.DataFrame:
    arr = df.to_numpy(dtype=np.float32)
    smoothed = np.zeros_like(arr)
    smoothed[0] = arr[0]
    for t in range(1, len(arr)):
        smoothed[t] = alpha * arr[t] + (1 - alpha) * smoothed[t-1]
    slope = np.diff(smoothed, axis=0, prepend=smoothed[:1])
    cols = [f"slope_{i}" for i in range(slope.shape[1])]
    return pd.DataFrame(slope, columns=cols)
    
def spatial_contrast(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.to_numpy(dtype=np.float32)
    contrast = arr.max(axis=1) - arr.min(axis=1)
    return pd.DataFrame({"spatial_contrast": contrast})

def rolling_std(df: pd.DataFrame, window: int = 150) -> pd.DataFrame:
    return df.rolling(window=window, min_periods=1).std().add_prefix("std_")


