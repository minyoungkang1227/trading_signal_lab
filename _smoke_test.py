"""
샌드박스 환경은 외부 시세 API(업비트/KRX)에 대한 네트워크 접근이 막혀 있어,
합성 OHLCV 데이터로 전체 파이프라인(지표 계산 -> 이벤트 스터디 -> 요약/플롯)에
런타임 오류가 없는지만 검증하는 스모크 테스트. 실제 사용자는 run_backtest.py로
진짜 업비트/KRX 데이터를 받아서 돌리면 된다.
"""
import numpy as np
import pandas as pd

from indicators import institutional_displacement, delta_trading, big_sales, ultimate_rsi, vp_box, momentum
from backtest.engine import compare_all, SIGNAL_SPECS

np.random.seed(42)
n = 1500
dates = pd.date_range("2022-01-01", periods=n, freq="D")
ret = np.random.normal(0, 0.02, n)
close = 100 * np.exp(np.cumsum(ret))
high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
open_ = close * (1 + np.random.normal(0, 0.005, n))
volume = np.abs(np.random.normal(1000, 400, n)) + (np.abs(ret) * 50000)

df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates)
df.index.name = "date"

dfs = {
    "institutional_displacement": institutional_displacement.compute(df),
    "institutional_displacement_z": institutional_displacement.compute(df, vol_mode="zscore"),
    "delta_trading": delta_trading.compute(df),
    "big_sales": big_sales.compute(df),
    "ultimate_rsi": ultimate_rsi.compute(df),
    "vp_box": vp_box.compute(df),
}

for name, d in dfs.items():
    print(name, "->", d.shape, [c for c in d.columns if c not in df.columns])

summary, results = compare_all(dfs)
print("\nsummary shape:", summary.shape)
print(summary.head(20))

for label, res in results.items():
    print(label, "n_events=", res.n_events)

assert not summary.empty, "summary가 비어있으면 안 됨"

# 모멘텀 필터 경로도 확인
mom_df = momentum.compute(df, lookback=60, window=120)
assert {"mom_return", "mom_percentile"} <= set(mom_df.columns)
filtered_dfs = momentum.apply_filter(dfs, mom_df, SIGNAL_SPECS, threshold=0.7)
mom_summary, mom_results = compare_all(filtered_dfs)
print("\n[모멘텀 필터 적용 후] summary shape:", mom_summary.shape)
for label, res in mom_results.items():
    print(f"  {label}: n_events={res.n_events}")
# 필터는 이벤트 수를 줄이거나 같게만 만들어야 한다(늘어나면 로직 버그)
for label in results:
    if label in mom_results:
        assert mom_results[label].n_events <= results[label].n_events, \
            f"{label}: 모멘텀 필터 후 이벤트가 오히려 늘어남 (버그 의심)"

print("\nSMOKE TEST OK")
