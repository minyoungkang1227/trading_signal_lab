"""
Amihud(2002), "Illiquidity and Stock Returns: Cross-Section and Time-Series
Effects"의 비유동성(illiquidity) 척도를 파이프라인 두 곳에 쓴다.

원 논문 정의: ILLIQ_t = |R_t| / DVOL_t (그날의 |수익률|을 그날의 거래대금(달러
거래량)으로 나눈 값). 값이 클수록 "적은 거래대금으로도 가격이 크게 움직인다"는
뜻이라 비유동적이다. 원 논문은 이 값을 종목별로 월평균 내서 여러 종목 간
횡단면 비교(비유동적인 종목일수록 기대수익률이 높다는 프리미엄 검증)에 쓰지만,
여기서는 momentum.py와 같은 방식으로 종목 하나의 시계열 안에서만 쓴다 — Amihud
비율의 절대적 크기는 통화 단위·가격대·거래관행에 따라 천차만별이라(원화 vs
KRW코인, 대형주 vs 소형주) 종목 간 비교가 무의미하므로, "지금이 이 종목 자기
자신의 역사에서 유동성이 좋은 편인가 나쁜 편인가"라는 시계열 퍼센타일로만
정규화해서 쓴다.

두 가지 역할:
1. apply_filter() — [2]단계 이벤트 필터. 유동성이 너무 낮은(=비유동성 퍼센타일이
   너무 높은) 시점에 뜬 신호는, 실제로는 그만한 물량을 그 가격에 체결할 수 없어
   이벤트 스터디상 유의해 보여도 현실성이 떨어진다. 이런 구간의 이벤트를 걸러낸다.
2. estimate_slippage_bps() — [3]단계 슬리피지 동적 추정. 지금까지 --slippage-bps는
   전 구간에 동일한 고정값을 썼는데, 실제로는 유동성이 낮을수록 체결 시 슬리피지가
   커진다. 이 비유동성 퍼센타일에 비례해 매 이벤트(봉)마다 슬리피지 bp를 다르게
   추정한다.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame, window: int = 20, percentile_window: int = 252,
            close_col: str = "close", volume_col: str = "volume") -> pd.DataFrame:
    """
    window            : 일별 비유동성(daily_illiq)을 평활화할 롤링 윈도우(봉수).
                         원 논문의 "월별 평균"에 대응하는 스무딩 — 하루짜리
                         이상치(거래정지 직후 급등락 등)에 흔들리지 않게 한다.
                         기본 20봉≈1개월.
    percentile_window : amihud를 그 종목 자기 자신의 과거 분포 대비 퍼센타일로
                         정규화할 롤링 윈도우(봉수). 기본 252봉≈1년.
    close_col, volume_col : OHLCV 컬럼명. 거래대금은 close*volume으로 근사한다
                         (원화/코인 거래대금을 따로 안 주는 데이터 소스가 많아서).

    반환: df에 다음 컬럼을 추가한 복사본.
      dollar_volume     : close*volume (그날의 거래대금 근사)
      daily_illiq       : |당일수익률| / dollar_volume. dollar_volume<=0이거나
                          0으로 나눠 inf가 되는 날은 NaN 처리.
      amihud            : daily_illiq의 window봉 롤링 평균.
      amihud_percentile : amihud를 percentile_window봉 롤링 윈도우에서 퍼센타일
                          랭크(0~1)한 값. 1에 가까울수록 "이 종목 역사상 가장
                          비유동적이었던 시기에 가깝다"는 뜻. 윈도우가 아직 덜
                          찬 구간은 NaN.
    """
    out = df.copy()
    ret = out[close_col].pct_change()
    dollar_volume = out[close_col] * out[volume_col]

    daily_illiq = (ret.abs() / dollar_volume).replace([np.inf, -np.inf], np.nan)
    daily_illiq = daily_illiq.where(dollar_volume > 0)

    amihud = daily_illiq.rolling(window, min_periods=max(2, window // 2)).mean()
    amihud_percentile = amihud.rolling(
        percentile_window, min_periods=max(10, percentile_window // 4)
    ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    out["dollar_volume"] = dollar_volume
    out["daily_illiq"] = daily_illiq
    out["amihud"] = amihud
    out["amihud_percentile"] = amihud_percentile
    return out


def apply_filter(dfs_by_indicator: dict[str, pd.DataFrame], liq_df: pd.DataFrame,
                  signal_specs, threshold: float = 0.9) -> dict[str, pd.DataFrame]:
    """
    dfs_by_indicator : {"institutional_displacement": df, ...} (indicators.*.compute() 결과)
    liq_df           : compute()의 반환값(amihud_percentile 컬럼 포함, 같은 df 기반
                       이어야 인덱스가 맞음)
    signal_specs     : backtest.engine.SIGNAL_SPECS 형식의
                        (indicator_key, event_col, direction, label) 리스트
    threshold        : 0~1. amihud_percentile이 이 값을 "초과"하는(=가장 비유동적인
                        상위 (1-threshold) 구간) 시점의 이벤트는 제외한다. 기본
                        0.9 = 이 종목 역사상 가장 비유동적이었던 상위 10% 구간 제외.

    momentum.apply_filter()와 동일한 패턴 — 방향(롱/숏)과 무관하게 모든 이벤트
    컬럼에 같은 유동성 마스크를 AND 적용한다(비유동성은 방향성이 없는 문제이므로).
    윈도우가 덜 차서 amihud_percentile을 계산 못한(NaN) 구간은 보수적으로
    "통과"시킨다 — 정보가 없다고 이벤트를 없애버리면 초반 표본이 부당하게
    줄어들기 때문.
    """
    liquid_enough = liq_df["amihud_percentile"] <= threshold
    liquid_enough = liquid_enough | liq_df["amihud_percentile"].isna()

    filtered = {k: v.copy() for k, v in dfs_by_indicator.items()}
    for indicator_key, event_col, _direction, _label in signal_specs:
        df = filtered.get(indicator_key)
        if df is None or event_col not in df.columns:
            continue
        gate = liquid_enough.reindex(df.index).fillna(True)
        df[event_col] = df[event_col].fillna(False) & gate

    return filtered


def estimate_slippage_bps(liq_df: pd.DataFrame, base_bps: float = 5.0,
                           max_extra_bps: float = 50.0) -> pd.Series:
    """
    amihud_percentile(0~1)을 [base_bps, base_bps+max_extra_bps] 구간으로 선형
    매핑해 봉(날짜)별 슬리피지(bp) 추정치를 만든다. percentile이 NaN(윈도우 부족
    등으로 계산 불가)이면 0으로 간주해 base_bps만 적용한다 — 정보가 없다고
    슬리피지를 아예 0으로 두면 위험 측이 아니라 낙관적인 쪽으로 치우치므로,
    최소한 base_bps는 항상 반영되게 한다.

    반환: df.index와 같은 인덱스를 가진 pd.Series(bp 단위). backtest.engine의
    forward_return()/event_study()/compare_all() 등은 slippage_bps 인자로 스칼라
    대신 이 Series를 그대로 받을 수 있다(날짜별로 다른 슬리피지를 적용).
    """
    pct = liq_df["amihud_percentile"].fillna(0.0).clip(0.0, 1.0)
    return base_bps + max_extra_bps * pct
