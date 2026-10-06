"""
Pine Script 원본: 'Big Sales'
"최근 length봉 동안 상승(또는 하락) 마감한 봉의 거래량 합이 0(=그 방향이 단 한 번도
없었음)이면서, 현재 봉 거래량이 평균 대비 급등"인 경우를 이벤트로 잡는다.

반환 컬럼:
  bull_event, bear_event : bool
  bull_tier, bear_tier    : 1(tiny)~5(huge), 이벤트가 아니면 0
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame, length: int = 7) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    volume = out["volume"]

    up_day = close > close.shift(1)
    down_day = close < close.shift(1)
    up_vol_sum = volume.where(up_day, 0.0).rolling(length).sum()
    down_vol_sum = volume.where(down_day, 0.0).rolling(length).sum()

    sma = volume.rolling(length).mean()
    thresholds = [2.00, 1.75, 1.50, 1.25, 1.00]  # f1..f5, 강한 순서

    no_up_in_window = up_vol_sum == 0
    no_down_in_window = down_vol_sum == 0

    bull_tier = pd.Series(0, index=out.index)
    bear_tier = pd.Series(0, index=out.index)
    for tier, mult in zip(range(5, 0, -1), thresholds):
        spike = volume > sma * mult
        bull_tier = np.where((bull_tier == 0) & no_up_in_window & spike, tier, bull_tier)
        bear_tier = np.where((bear_tier == 0) & no_down_in_window & spike, tier, bear_tier)

    out["bull_tier"] = bull_tier
    out["bear_tier"] = bear_tier
    out["bull_event"] = out["bull_tier"] > 0
    out["bear_event"] = out["bear_tier"] > 0
    return out
