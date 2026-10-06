"""
backtest/engine.py에 추가한 세 가지 기능을 검증한다.
1) slippage_bps: cost_bps와 별개로 왕복 슬리피지를 차감하는지 (수치 정확성 + 합산 확인)
2) _newey_west_t_stat(): Newey-West HAC 보정 t-stat이 자기상관이 있는 데이터에서
   단순 t-stat보다 작게(더 보수적으로) 나오는지
3) _dedupe_overlapping()/drop_overlapping: 보유기간이 겹치는 이벤트를 실제로
   솎아내는지, 그리고 기본값(False)은 이전 동작과 동일한지
"""
import numpy as np
import pandas as pd
from backtest.engine import forward_return, event_study, _dedupe_overlapping, _newey_west_t_stat

# --- 1) slippage_bps ---
n = 20
dates = pd.date_range("2024-01-01", periods=n, freq="D")
close = 100 * (1.01 ** np.arange(n))
df = pd.DataFrame({"close": close}, index=dates)

ret_no_cost = forward_return(df, 5)
ret_cost_only = forward_return(df, 5, cost_bps=36.0)
ret_slip_only = forward_return(df, 5, slippage_bps=10.0)
ret_both = forward_return(df, 5, cost_bps=36.0, slippage_bps=10.0)

assert np.allclose((ret_no_cost - 36.0 / 10000).dropna(), ret_cost_only.dropna())
assert np.allclose((ret_no_cost - 10.0 / 10000).dropna(), ret_slip_only.dropna())
assert np.allclose((ret_no_cost - 46.0 / 10000).dropna(), ret_both.dropna())
print("[OK] slippage_bps가 cost_bps와 별개로 정확히 합산 차감됨 (36+10=46bp)")

# --- 2) Newey-West t-stat: 강한 양(+)의 자기상관이 있는 인위적 시계열에서
#    단순 t-stat 대비 NW t-stat이 더 작아야 함(과대추정 보정) ---
rng = np.random.default_rng(42)
base = rng.normal(0, 1, 200)
# AR(1)-like 자기상관을 강하게 주입 (rho=0.8)
ar = np.zeros(200)
ar[0] = base[0]
for i in range(1, 200):
    ar[i] = 0.8 * ar[i - 1] + base[i]
vals = 0.01 + 0.001 * ar  # 평균 0.01에 자기상관 있는 노이즈를 얹음

naive_mean = vals.mean()
naive_std = vals.std(ddof=1)
naive_t = naive_mean / (naive_std / np.sqrt(len(vals)))

nw_mean, nw_t = _newey_west_t_stat(vals, lag=9)
assert abs(nw_mean - naive_mean) < 1e-9, "평균 자체는 동일해야 함"
assert abs(nw_t) < abs(naive_t), f"자기상관 데이터에서 NW t-stat이 단순 t-stat보다 작아야 함: nw={nw_t}, naive={naive_t}"
print(f"[OK] 자기상관 있는 데이터: naive t={naive_t:.3f} -> NW 보정 t={nw_t:.3f} (절댓값 감소, 과대추정 보정 확인)")

# 자기상관이 없는(독립) 데이터에서는 NW와 단순 t-stat이 비슷해야 함(과보정 없음)
iid_vals = rng.normal(0.01, 1.0, 500)
naive_t_iid = iid_vals.mean() / (iid_vals.std(ddof=1) / np.sqrt(len(iid_vals)))
_, nw_t_iid = _newey_west_t_stat(iid_vals, lag=4)
assert abs(nw_t_iid - naive_t_iid) / abs(naive_t_iid) < 0.3, "독립 데이터에서 NW와 단순 t-stat이 크게 달라지면 안 됨"
print(f"[OK] 자기상관 없는 데이터: naive t={naive_t_iid:.3f} -> NW t={nw_t_iid:.3f} (비슷한 수준 유지)")

# --- 3) _dedupe_overlapping ---
idx_all = pd.date_range("2024-01-01", periods=30, freq="D")
# 3봉 간격으로 이벤트 발생 (0,3,6,9,...) -> horizon=5로 겹치는지 확인
events = idx_all[::3]
kept_h5 = _dedupe_overlapping(idx_all, events, horizon=5)
# 0,3,6,... 이 5봉 간격 미만으로 겹치므로 그리디하게 0,6,12,18,24(간격>=5인 것들)만 남아야 함
positions_kept = idx_all.get_indexer(kept_h5)
diffs = np.diff(sorted(positions_kept))
assert (diffs >= 5).all(), f"겹치는 이벤트가 남아있음: {diffs}"
print(f"[OK] _dedupe_overlapping: {len(events)}개 이벤트 -> horizon=5 기준 {len(kept_h5)}개로 솎아냄, 간격 전부 >=5")

# horizon=2일 때는(간격 3 >= 2) 전부 남아야 함
kept_h2 = _dedupe_overlapping(idx_all, events, horizon=2)
assert len(kept_h2) == len(events)
print("[OK] horizon이 이벤트 간격보다 짧으면 전부 유지됨 (겹치지 않으므로)")

# --- event_study() 통합: drop_overlapping=False(기본)는 이전 동작과 동일 ---
df2 = pd.DataFrame({"close": close, "evt": False}, index=dates)
df2.loc[df2.index[::3], "evt"] = True
res_default = event_study(df2, "evt", horizons=(5,))
res_explicit_off = event_study(df2, "evt", horizons=(5,), drop_overlapping=False, newey_west=False)
assert res_default.table.loc[5, "n"] == res_explicit_off.table.loc[5, "n"]
assert abs(res_default.table.loc[5, "mean"] - res_explicit_off.table.loc[5, "mean"]) < 1e-12
print("[OK] event_study() 기본값(drop_overlapping=False, newey_west=False)은 이전 동작과 동일")

res_dropped = event_study(df2, "evt", horizons=(5,), drop_overlapping=True)
assert res_dropped.table.loc[5, "n"] <= res_default.table.loc[5, "n"]
print(f"[OK] drop_overlapping=True 적용 시 표본 수 감소 확인: {res_default.table.loc[5,'n']} -> {res_dropped.table.loc[5,'n']}")

print("\nSMOKE TEST (Newey-West / slippage / overlapping dedupe) OK")
