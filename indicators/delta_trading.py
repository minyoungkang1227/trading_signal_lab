"""
Pine Script 원본: '[Delta Trading]' (ⓒ hapharmonic)의 EMA 크로스 + 거래량 확인 신호 부분만 재구현.
(캔들 강조/테이블 시각화는 백테스트와 무관하므로 제외)

반환 컬럼:
  crossover_confirmed, crossunder_confirmed : bool
  buy_vol, sell_vol : Close Location Value 근사 매수/매도 거래량 (참고용)
"""
from __future__ import annotations
import pandas as pd


def _is_volume_confirmed(volume: pd.Series, length: int) -> pd.Series:
    ma = volume.rolling(length).mean()
    up_sum = volume.where(volume > ma, 0.0).rolling(length).sum()
    down_sum = volume.where(volume < ma, 0.0).rolling(length).sum()
    return up_sum >= down_sum


def compute(df: pd.DataFrame, ema_fast: int = 12, ema_slow: int = 26,
            vol_confirm_len: int = 6, use_volume_confirmation: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=ema_fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=ema_slow, adjust=False).mean()

    prev_diff = (out["ema_fast"] - out["ema_slow"]).shift(1)
    curr_diff = out["ema_fast"] - out["ema_slow"]
    crossover = (prev_diff <= 0) & (curr_diff > 0)
    crossunder = (prev_diff >= 0) & (curr_diff < 0)

    isv = _is_volume_confirmed(out["volume"], vol_confirm_len)
    out["crossover_confirmed"] = crossover & ((~use_volume_confirmation) | isv)
    out["crossunder_confirmed"] = crossunder & ((~use_volume_confirmation) | isv)

    rng = (out["high"] - out["low"]).replace(0, pd.NA)
    out["buy_vol"] = (out["volume"] * (out["close"] - out["low"]) / rng).fillna(out["volume"] / 2)
    out["sell_vol"] = out["volume"] - out["buy_vol"]
    return out
