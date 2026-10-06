"""
Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers:
Implications for Stock Market Efficiency"의 모멘텀(추세지속) 논리를,
기존 5개 지표의 이벤트에 대한 사전 필터로 사용한다.

원 논문 방법론: 매달 전체 유니버스를 과거 J개월(3~12개월) 수익률로
횡단면(cross-sectional) 정렬해서 상위 10%(승자)/하위 10%(패자) 포트폴리오를
구성하고, K개월(3~12개월) 보유한다. 승자는 계속 승자, 패자는 계속 패자인
경향(추세지속)이 유의하게 관측됐다는 게 핵심 결과.

이 랩과의 차이(근사): 여기서는 종목 하나씩 시계열로 백테스트하므로 원 논문의
횡단면 정렬이 불가능하다. 대신 "그 종목 자신의 과거 window 기간 동안의
모멘텀 분포 대비 지금이 몇 퍼센타일인가"라는 시계열(time-series) 근사로
대체했다 — 실제 다른 종목들과 비교한 승자/패자 그룹과는 다른 근사치이므로,
엄밀한 재현이 아니라 방향성 필터로만 사용할 것.

사용 로직: bull/long류 이벤트(EMA 골든크로스, 기관캔들 강세 등)는 "이미
상승 모멘텀이 강한 구간"에서 떴을 때만, bear/short류 이벤트는 "이미 하락
모멘텀이 강한 구간"에서 떴을 때만 인정한다. 즉 지금의 지표 신호가 추세
'전환'이 아니라 추세 '지속'의 초입을 잡고 있는지를 검증하는 필터다.

반환 컬럼:
  mom_return     : lookback봉 전 대비 수익률
  mom_percentile : 그 수익률을 과거 window봉 동안의 분포에서 순위 매긴 퍼센타일(0~1)
"""
from __future__ import annotations
import pandas as pd


def compute(df: pd.DataFrame, lookback: int = 126, window: int = 252) -> pd.DataFrame:
    """
    lookback : 모멘텀을 측정할 과거 구간(봉수). 일봉 기준 126봉 ≈ 6개월
               (원 논문에서 가장 성과가 좋았던 J=6 근사).
    window   : 퍼센타일을 매길 롤링 윈도우(봉수). 일봉 기준 252봉 ≈ 1년.
    """
    out = df.copy()
    mom_return = out["close"] / out["close"].shift(lookback) - 1
    mom_percentile = mom_return.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    out["mom_return"] = mom_return
    out["mom_percentile"] = mom_percentile
    return out


def apply_filter(dfs_by_indicator: dict[str, pd.DataFrame], mom_df: pd.DataFrame,
                  signal_specs, threshold: float = 0.7) -> dict[str, pd.DataFrame]:
    """
    dfs_by_indicator : {"institutional_displacement": df, ...} (indicators.*.compute() 결과)
    mom_df           : compute()의 반환값 (mom_percentile 컬럼 포함)
    signal_specs     : backtest.engine.SIGNAL_SPECS 형식의
                        (indicator_key, event_col, direction(+1/-1), label) 리스트
    threshold        : 0.5~1.0. 0.7이면 "과거 window 내 상위 30% 모멘텀"에서만
                        롱(bull) 이벤트를, "하위 30%"에서만 숏(bear) 이벤트를 인정.

    각 지표 df를 복사한 뒤, direction=+1(롱) 이벤트 컬럼은
    mom_percentile >= threshold 조건과, direction=-1(숏) 이벤트 컬럼은
    mom_percentile <= 1-threshold 조건과 AND해서 덮어쓴 새 dict를 반환한다.
    """
    pct = mom_df["mom_percentile"]
    filtered = {k: v.copy() for k, v in dfs_by_indicator.items()}

    for indicator_key, event_col, direction, _label in signal_specs:
        df = filtered.get(indicator_key)
        if df is None or event_col not in df.columns:
            continue
        aligned_pct = pct.reindex(df.index)
        gate = (aligned_pct >= threshold) if direction == 1 else (aligned_pct <= (1 - threshold))
        df[event_col] = df[event_col].fillna(False) & gate.fillna(False)

    return filtered
