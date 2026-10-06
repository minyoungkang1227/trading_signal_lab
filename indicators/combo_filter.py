"""
Pine Script 원본: 직접 설계한 결합 진입 필터
'Candle+Volume Entry Filter (mine)' (pinescript-candle-signal/my_filter/
candle_volume_entry_filter.pine, 저작자 강민영).

reference/ 폴더의 공개 지표(거래량 스파이크 캔들 탐지, 캔들 내부 매수/매도 델타,
RSI 기반 추세 필터)를 참고해, 원본과 달리 세 요소를 OR/AND가 아니라 연속값으로
결합한 지표. 다른 5개 지표(institutional_displacement 등)와 달리 이진 신호가
아니라 0~1 "구성 점수(composite_score)"를 먼저 만들고, 그 점수가 threshold를
넘는지로 최종 신호를 낸다.

1. 캔들 구조 스코어: 몸통(body)이 전체 고가-저가 범위에서 차지하는 비율(body_pct)과
   진행 방향 반대쪽 꼬리 비율(opp_wick_pct)로 "형태적 신뢰도"를 계산.
2. 거래량 델타 스코어: 평균 거래량 대비 스파이크 배율을 0~1로 정규화.
3. 추세 필터: 이동평균(SMA/EMA/WMA) 대비 캔들 진행 방향이 일치하는지.
4. 결합: 구조·거래량 스코어를 가중합(0.5/0.5)한 뒤, 추세 정렬 여부를 곱셈
   마스크로 적용(추세 역행 시 점수를 0으로 무효화) — 단순 AND가 아니라 연속
   점수로 만든 게 원본 설계의 핵심.

원본(Pine)과의 차이: 전부 그대로 재구현했고 근사는 없음(ta.sma/ta.ema/ta.wma,
거래량 sma, 구조·거래량 스코어 계산식 전부 Pine 코드와 1:1 대응). 원본의
score 임계값(0.62)은 TradingView 차트 백테스트로 0.50~0.75 구간을 스캔해
승률·손익비가 가장 안정적인 지점으로 채택한 값 — 이 파이프라인의
event_study/judge_indicator로 재검증 전까지는 "참고용 채택값"으로만 취급할 것.

반환 컬럼:
  body_pct, opp_wick_pct       : 캔들 구조 중간값(%)
  structure_score, vol_score, composite_score : 0~1 연속 점수
  trend_aligned                : bool (캔들 방향과 추세 방향 일치 여부)
  bull_signal, bear_signal     : bool (원본의 longSignal/shortSignal)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _trend_ma(close: pd.Series, length: int, ma_type: str) -> pd.Series:
    if ma_type == "SMA":
        return close.rolling(length).mean()
    if ma_type == "EMA":
        return close.ewm(span=length, adjust=False).mean()
    if ma_type == "WMA":
        weights = np.arange(1, length + 1, dtype=float)
        return close.rolling(length).apply(
            lambda x: float((x * weights).sum() / weights.sum()), raw=True
        )
    raise ValueError(f"unknown ma_type: {ma_type!r} (SMA/EMA/WMA 중 하나여야 함)")


def compute(df: pd.DataFrame, body_min_pct: float = 55.0, wick_max_pct: float = 25.0,
            vol_len: int = 20, vol_mult: float = 1.6, ma_len: int = 50,
            ma_type: str = "EMA", score_threshold: float = 0.62) -> pd.DataFrame:
    """
    body_min_pct   : 몸통이 전체 범위에서 차지하는 최소 비율(%). 원본 기본값 55.
    wick_max_pct   : 진행 방향 반대쪽 꼬리의 최대 허용 비율(%). 원본 기본값 25.
    vol_len        : 거래량 SMA 길이. 원본 기본값 20.
    vol_mult       : 평균 거래량 대비 이 배수 이상일 때만 "스파이크"로 인정. 원본 기본값 1.6.
    ma_len, ma_type: 추세 필터 이동평균 길이·종류(SMA/EMA/WMA). 원본 기본값 50, EMA.
    score_threshold: composite_score가 이 값 이상이어야 최종 신호로 인정. 원본 채택값 0.62.
    """
    out = df.copy()

    rng = out["high"] - out["low"]
    rng_safe = rng.replace(0, np.nan)
    body = (out["close"] - out["open"]).abs()
    up_wick = out["high"] - out[["close", "open"]].max(axis=1)
    dn_wick = out[["close", "open"]].min(axis=1) - out["low"]
    bullish = out["close"] > out["open"]

    body_pct = (body / rng_safe * 100.0).fillna(0.0)
    opp_wick = pd.Series(np.where(bullish, dn_wick, up_wick), index=out.index)
    opp_wick_pct = (opp_wick / rng_safe * 100.0).fillna(0.0)

    structure_ok = (body_pct >= body_min_pct) & (opp_wick_pct <= wick_max_pct)
    structure_score = (body_pct / 100.0).clip(upper=1.0)

    vol_sma = out["volume"].rolling(vol_len).mean()
    vol_sma_safe = vol_sma.replace(0, np.nan)
    vol_spike = (out["volume"] >= vol_sma * vol_mult) & (vol_sma > 0)
    vol_score = ((out["volume"] / (vol_sma_safe * vol_mult)).clip(upper=1.5) / 1.5).fillna(0.0)

    trend_ma = _trend_ma(out["close"], ma_len, ma_type)
    trend_up = out["close"] > trend_ma
    trend_down = out["close"] < trend_ma
    trend_aligned = pd.Series(np.where(bullish, trend_up, trend_down), index=out.index)
    # trend_ma가 아직 계산 안 된(NaN) 구간은 추세 판정 불가 -> 정렬 안 됨으로 보수 처리
    trend_aligned = trend_aligned.where(trend_ma.notna(), other=False)

    composite_score = (structure_score * 0.5 + vol_score * 0.5) * trend_aligned.astype(float)

    bull_signal = (bullish & structure_ok & vol_spike & trend_aligned
                   & (composite_score >= score_threshold)).fillna(False)
    bear_signal = ((~bullish) & structure_ok & vol_spike & trend_aligned
                   & (composite_score >= score_threshold)).fillna(False)

    out["body_pct"] = body_pct
    out["opp_wick_pct"] = opp_wick_pct
    out["structure_score"] = structure_score
    out["vol_score"] = vol_score
    out["trend_aligned"] = trend_aligned
    out["composite_score"] = composite_score
    out["bull_signal"] = bull_signal
    out["bear_signal"] = bear_signal
    return out
