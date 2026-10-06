"""
Pine Script 원본: 'Filter RSI' (Ultimate RSI, ⓒ RICH TRADING SCHOOL)
최근 length봉의 신고점/신저점을 갱신하는 순간에는 구간 전체 레인지를 모멘텀으로,
그 외에는 평범한 종가 변화량을 모멘텀으로 써서 정규 RSI보다 레인지 돌파에 민감하게
반응하도록 만든 오실레이터.

반환 컬럼:
  arsi, signal : 0~100 오실레이터 값
  bull_signal  : arsi가 과매도(osValue) 구간에서 위로 이탈 (반등 신호)
  bear_signal  : arsi가 과매수(obValue) 구간에서 아래로 이탈 (반락 신호)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _ma(series: pd.Series, length: int, method: str) -> pd.Series:
    if method == "EMA":
        return series.ewm(span=length, adjust=False).mean()
    if method == "SMA":
        return series.rolling(length).mean()
    if method == "RMA":
        return series.ewm(alpha=1 / length, adjust=False).mean()
    if method == "TMA":
        return series.rolling(length).mean().rolling(length).mean()
    raise ValueError(f"unknown ma method: {method}")


def compute(df: pd.DataFrame, length: int = 14, method1: str = "RMA",
            smooth: int = 14, method2: str = "EMA",
            ob_value: float = 80.0, os_value: float = 20.0) -> pd.DataFrame:
    out = df.copy()
    src = out["close"]

    upper = src.rolling(length).max()
    lower = src.rolling(length).min()
    r = upper - lower
    d = src.diff()

    new_high = upper > upper.shift(1)
    new_low = lower < lower.shift(1)
    diff = np.where(new_high, r, np.where(new_low, -r, d))
    diff = pd.Series(diff, index=out.index)

    num = _ma(diff, length, method1)
    den = _ma(diff.abs(), length, method1)
    arsi = (num / den) * 50 + 50
    signal = _ma(arsi, smooth, method2)

    out["arsi"] = arsi
    out["signal"] = signal
    prev = arsi.shift(1)
    out["bull_signal"] = (prev <= os_value) & (arsi > os_value)
    out["bear_signal"] = (prev >= ob_value) & (arsi < ob_value)
    return out
