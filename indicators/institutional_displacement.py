"""
Pine Script 원본: 'Institutional Displacement & Volume Delta'

원본은 request.security_lower_tf로 인트라바 매수/매도 거래량을 실측하지만,
여기서는 봉 단위 OHLCV만으로 백테스트하므로 그 부분은 근사(Close Location Value)로
대체한다. 신호 자체(거래량 급등 + 몸통 비율 조건)는 원본과 동일한 로직이다.

거래량 이상치 판정 방식은 두 가지를 선택할 수 있다 (vol_mode):
  - "multiple" (기본, 원본과 동일): 거래량 > 이동평균 × vol_mult
  - "zscore"  : Kyle(1985) 모형에서 유도한 통계적 기준. 주문흐름 y의
                무조건부 표준편차가 sqrt(2)*sigma_u 라는 결과를 이용해,
                거래량을 그 종목 고유의 변동성(rolling std) 대비 z-스코어로
                표준화한 뒤 z > z_threshold(기본 1.96, 95% 유의수준)로 판정한다.
                "평균의 몇 배"라는 임의적 배수 대신, 종목마다 다른 노이즈
                수준(σᵤ)을 반영한 기준이 된다.

반환 컬럼:
  bull_shift, bear_shift : bool  (원본의 bullShift/bearShift와 동일 조건)
  body_ratio, high_vol   : 참고용 중간값
  vol_z                  : vol_mode="zscore"일 때만 채워지는 거래량 z-스코어
"""
from __future__ import annotations
import pandas as pd


def compute(df: pd.DataFrame, vol_len: int = 20, vol_mult: float = 2.0,
            body_pct: float = 50.0, vol_mode: str = "multiple",
            z_threshold: float = 1.96) -> pd.DataFrame:
    out = df.copy()
    avg_vol = out["volume"].rolling(vol_len).mean()

    if vol_mode == "zscore":
        std_vol = out["volume"].rolling(vol_len).std()
        vol_z = (out["volume"] - avg_vol) / std_vol.replace(0, pd.NA)
        vol_z = vol_z.fillna(0.0)
        out["vol_z"] = vol_z
        high_vol = vol_z > z_threshold
    elif vol_mode == "multiple":
        high_vol = out["volume"] > avg_vol * vol_mult
    else:
        raise ValueError(f"unknown vol_mode: {vol_mode!r} (multiple 또는 zscore)")

    candle_range = out["high"] - out["low"]
    candle_body = (out["close"] - out["open"]).abs()
    body_ratio = (candle_body / candle_range.replace(0, pd.NA)).fillna(0.0)
    is_displacement = body_ratio >= (body_pct / 100)

    out["high_vol"] = high_vol
    out["body_ratio"] = body_ratio
    out["bull_shift"] = (out["close"] > out["open"]) & high_vol & is_displacement
    out["bear_shift"] = (out["close"] < out["open"]) & high_vol & is_displacement
    return out
