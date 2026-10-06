"""
Pine Script 원본: 'VP BOX indicator'
상위 타임프레임(기본 주봉) 구간마다 가격대별 거래량(볼륨 프로파일)을 계산해
가장 거래량이 몰린 가격대(POC)를 뽑는다. 원본은 실시간으로 그리지만, 여기서는
"완료된 이전 구간의 POC"를 다음 구간에 지지/저항 레벨로 넘겨주는 방식으로
백테스트용 신호를 만든다.

반환 컬럼:
  prev_poc : 직전에 완료된 구간의 POC 가격 (해당 구간 동안 전부 동일값, NaN=아직 없음)
  poc_support_test    : 가격이 prev_poc까지 내려왔다가 그 위로 마감 (지지 확인)
  poc_resistance_test : 가격이 prev_poc까지 올라왔다가 그 아래로 마감 (저항 확인)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _period_poc(period_df: pd.DataFrame, bins: int) -> float | None:
    high = period_df["high"].max()
    low = period_df["low"].min()
    if not np.isfinite(high) or not np.isfinite(low) or high <= low:
        return None
    edges = np.linspace(low, high, bins + 1)
    freq = np.zeros(bins)
    idx = np.digitize(period_df["close"].values, edges) - 1
    idx = np.clip(idx, 0, bins - 1)
    for i, v in zip(idx, period_df["volume"].values):
        freq[i] += v
    poc_bin = int(np.argmax(freq))
    return float((edges[poc_bin] + edges[poc_bin + 1]) / 2)


def compute(df: pd.DataFrame, bins: int = 20, freq: str = "W") -> pd.DataFrame:
    out = df.copy()
    out["prev_poc"] = np.nan

    period_key = out.index.to_period(freq)
    pocs = {}
    for key, group in out.groupby(period_key):
        pocs[key] = _period_poc(group, bins)

    ordered_keys = sorted(pocs.keys())
    prev_poc_by_period = {}
    last_valid = None
    for key in ordered_keys:
        prev_poc_by_period[key] = last_valid
        if pocs[key] is not None:
            last_valid = pocs[key]

    out["prev_poc"] = period_key.map(prev_poc_by_period).astype(float)

    touched = (out["low"] <= out["prev_poc"]) & (out["high"] >= out["prev_poc"])
    out["poc_support_test"] = touched & (out["close"] > out["prev_poc"])
    out["poc_resistance_test"] = touched & (out["close"] < out["prev_poc"])
    return out
