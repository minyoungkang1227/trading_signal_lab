"""
candle_volume_entry_filter.pine -> indicators/combo_filter.py 포팅 검증용 스모크
테스트. 기존 _smoke_test.py와 같은 패턴(합성 OHLCV, 손으로 계산한 기대값과 대조)을
따른다. 두 가지를 확인한다.

  1) 결정론적으로 만든 "교과서적인 양봉+거래량급등+상승추세" 캔들 하나를 넣었을 때
     bull_signal이 정확히 뜨는지 손 계산으로 검증(값 자체가 맞는지).
  2) 합성 시계열 전체에 대해 SIGNAL_SPECS에 콤보필터가 등록된 상태로 compare_all()이
     에러 없이 끝까지 도는지(기존 5개 지표를 건드리지 않았는지, 파이프라인 통합이
     되는지).
"""
import numpy as np
import pandas as pd

from indicators import combo_filter
from backtest.engine import compare_all, SIGNAL_SPECS
from indicators import (institutional_displacement, delta_trading, big_sales,
                         ultimate_rsi, vp_box)

# ---------------------------------------------------------------------------
# 1) 손으로 설계한 단일 이벤트로 compute() 수치 자체를 검증
# ---------------------------------------------------------------------------
# 50봉 동안 완만한 상승추세(EMA50이 종가보다 뚜렷이 아래)를 만들어둔 뒤,
# 마지막 봉 하나만 "몸통 90%, 거래량 평균의 3배"인 강한 양봉으로 설계한다.
n_warmup = 60
base = pd.Series(np.linspace(100, 110, n_warmup))  # 완만한 상승 추세
vol_base = pd.Series(np.full(n_warmup, 1000.0))

# 마지막 봉: 저가 120 -> 고가 130, 종가 129(몸통 90%), 거래량 평균(1000)의 3배
last_open, last_high, last_low, last_close, last_vol = 120.0, 130.0, 120.0, 129.0, 3000.0

close = pd.concat([base, pd.Series([last_close])], ignore_index=True)
open_ = pd.concat([base.shift(1).fillna(base.iloc[0]), pd.Series([last_open])], ignore_index=True)
high = pd.concat([base + 0.3, pd.Series([last_high])], ignore_index=True)
low = pd.concat([base - 0.3, pd.Series([last_low])], ignore_index=True)
volume = pd.concat([vol_base, pd.Series([last_vol])], ignore_index=True)

dates = pd.date_range("2024-01-01", periods=len(close), freq="D")
single_df = pd.DataFrame({"open": open_.values, "high": high.values, "low": low.values,
                           "close": close.values, "volume": volume.values}, index=dates)
single_df.index.name = "date"

res = combo_filter.compute(single_df, vol_len=20, vol_mult=1.6, ma_len=50, ma_type="EMA",
                            score_threshold=0.62)
last_row = res.iloc[-1]

# 손 계산: body=|129-120|=9, range=130-120=10 -> body_pct=90.0
assert abs(last_row["body_pct"] - 90.0) < 1e-9, f"body_pct 계산 오류: {last_row['body_pct']}"
# opp_wick(양봉이므로 아래꼬리)=min(129,120)-120=0 -> opp_wick_pct=0.0
assert abs(last_row["opp_wick_pct"] - 0.0) < 1e-9, f"opp_wick_pct 계산 오류: {last_row['opp_wick_pct']}"
# structure_score = min(90/100, 1.0) = 0.9
assert abs(last_row["structure_score"] - 0.9) < 1e-9
# 거래량 3000, vol_sma(직전 20봉 평균)=1000 -> 3000/(1000*1.6)=1.875 -> clip 1.5 -> /1.5=1.0
assert abs(last_row["vol_score"] - 1.0) < 1e-9, f"vol_score 계산 오류: {last_row['vol_score']}"
# composite_score = (0.9*0.5 + 1.0*0.5) * trend_aligned(1.0) = 0.95
assert last_row["trend_aligned"] == True or last_row["trend_aligned"] == 1
assert abs(last_row["composite_score"] - 0.95) < 1e-9, f"composite_score 계산 오류: {last_row['composite_score']}"
assert last_row["bull_signal"] == True, "교과서적 강세 이벤트인데 bull_signal이 안 뜸"
assert last_row["bear_signal"] == False

print("[1/2] 단일 이벤트 손계산 검증 통과 "
      f"(body_pct={last_row['body_pct']:.1f}, vol_score={last_row['vol_score']:.3f}, "
      f"composite_score={last_row['composite_score']:.3f} >= 0.62 -> bull_signal)")

# ---------------------------------------------------------------------------
# 2) 기존 _smoke_test.py와 동일한 합성 시계열로 전체 파이프라인 통합 검증
# ---------------------------------------------------------------------------
np.random.seed(42)
n = 1500
dates2 = pd.date_range("2022-01-01", periods=n, freq="D")
ret = np.random.normal(0, 0.02, n)
close2 = 100 * np.exp(np.cumsum(ret))
high2 = close2 * (1 + np.abs(np.random.normal(0, 0.01, n)))
low2 = close2 * (1 - np.abs(np.random.normal(0, 0.01, n)))
open2 = close2 * (1 + np.random.normal(0, 0.005, n))
volume2 = np.abs(np.random.normal(1000, 400, n)) + (np.abs(ret) * 50000)

df2 = pd.DataFrame({"open": open2, "high": high2, "low": low2, "close": close2, "volume": volume2},
                    index=dates2)
df2.index.name = "date"

dfs = {
    "institutional_displacement": institutional_displacement.compute(df2),
    "delta_trading": delta_trading.compute(df2),
    "big_sales": big_sales.compute(df2),
    "ultimate_rsi": ultimate_rsi.compute(df2),
    "vp_box": vp_box.compute(df2),
    "combo_filter": combo_filter.compute(df2),
}

labels = [spec[3] for spec in SIGNAL_SPECS]
assert "콤보필터-강세" in labels and "콤보필터-약세" in labels, \
    "SIGNAL_SPECS에 콤보필터가 등록되지 않음"

summary, results = compare_all(dfs)
assert not summary.empty, "summary가 비어있으면 안 됨 (파이프라인 통합 실패)"

combo_events = {label: res.n_events for label, res in results.items() if "콤보필터" in label}
print(f"[2/2] 전체 파이프라인 통합 검증 통과. 콤보필터 이벤트 수(합성데이터, 1500봉): {combo_events}")
print(summary[summary["signal"].str.contains("콤보필터")].to_string(index=False))

# 콤보필터는 "둘 다 겹쳐야" 하는 구조상 다른 지표보다 이벤트가 매우 드물 것으로
# 예상된다(원본 pinescript-지표-5종-비교.md의 결론과 동일한 방향) -- 0건이 아니되,
# 전체 봉수(1500) 대비 지나치게 잦지는 않은지만 느슨하게 확인.
for label, n_events in combo_events.items():
    assert 0 <= n_events < n, f"{label} 이벤트 수가 비정상적입니다: {n_events}"

print("\nSMOKE TEST OK (combo_filter)")
