"""
backtest/engine.py의 거래비용(cost_bps)·체결 시점 지연(entry_lag/entry_col) 로직 검증.
결정적(등비수열) 가격 시계열로 forward_return()의 정확한 수치를 손으로 계산해
대조하고, 기본값(entry_lag=0, cost_bps=0)이 리팩터링 이전 동작과 완전히 동일한지도
함께 확인한다.
"""
import numpy as np
import pandas as pd
from backtest.engine import forward_return, event_study

# 가격이 매일 정확히 1%씩 오르는 완전 결정적 시계열로 entry_lag/cost_bps 정확성 검증
n = 30
dates = pd.date_range("2024-01-01", periods=n, freq="D")
close = 100 * (1.01 ** np.arange(n))
open_ = close / 1.005  # 시가는 종가보다 살짝 낮다고 가정(당일 상승분의 절반 지점)
df = pd.DataFrame({"close": close, "open": open_}, index=dates)

# 1) entry_lag=0, cost=0 -> 예전 동작과 동일해야 함: close[t+h]/close[t]-1
old_style = df["close"].shift(-5) / df["close"] - 1
new_style = forward_return(df, 5)
assert np.allclose(old_style.dropna(), new_style.dropna()), "entry_lag=0 default가 기존 동작과 다름"
print("[OK] entry_lag=0, cost=0 -> 기존 동작과 동일")

# 2) entry_lag=1, entry_col='open' -> exit=close[t+h], entry=open[t+1]
lagged = forward_return(df, 5, entry_lag=1, entry_col="open")
expected = df["close"].shift(-5) / df["open"].shift(-1) - 1
assert np.allclose(lagged.dropna(), expected.dropna())
print("[OK] entry_lag=1, entry_col='open' -> t+1 시가 진입 정확")

# 3) cost_bps 차감이 direction과 무관하게 항상 손실로 작동하는지 (숏 포지션에서도 비용은 비용)
long_ret = forward_return(df, 5, direction=1, cost_bps=36.0)
short_ret = forward_return(df, 5, direction=-1, cost_bps=36.0)
raw = df["close"].shift(-5) / df["close"] - 1
assert np.allclose(long_ret.dropna(), (raw * 1 - 36.0/10000).dropna())
assert np.allclose(short_ret.dropna(), (raw * -1 - 36.0/10000).dropna())
print("[OK] cost_bps는 direction과 무관하게 항상 차감됨(부호 반전 버그 없음)")

# 4) event_study()가 cost_bps 반영 시 무비용 대비 항상 낮은 평균수익률을 냄
df["evt"] = False
df.iloc[::3, df.columns.get_loc("evt")] = True
res_no_cost = event_study(df, "evt", horizons=(5,))
res_cost = event_study(df, "evt", horizons=(5,), entry_lag=1, entry_col="open", cost_bps=36.0)
assert res_cost.table.loc[5, "mean"] < res_no_cost.table.loc[5, "mean"]
print(f"[OK] event_study 비용 반영 시 평균수익률 감소: {res_no_cost.table.loc[5,'mean']:.5f} -> {res_cost.table.loc[5,'mean']:.5f}")

print("\nALL COST/ENTRY-LAG TESTS PASSED")
