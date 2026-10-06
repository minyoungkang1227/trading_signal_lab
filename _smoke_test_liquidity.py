"""
indicators/liquidity.py (Amihud(2002) 비유동성 척도) 검증.

1) compute(): 거래대금이 작고 가격변동이 큰 구간일수록 daily_illiq/amihud가
   커야 하고, amihud_percentile은 0~1 사이여야 한다.
2) apply_filter(): amihud_percentile이 threshold를 초과하는(가장 비유동적인)
   시점의 이벤트는 제외되고, NaN(윈도우 부족) 구간은 통과해야 한다.
3) estimate_slippage_bps(): percentile=0 -> base_bps, percentile=1 ->
   base_bps+max_extra_bps, 그 사이는 선형.
4) forward_return()/event_study()가 슬리피지를 스칼라 대신 pd.Series로 받았을 때
   봉마다 다른 슬리피지를 정확히 반영하는지.
"""
import numpy as np
import pandas as pd

from indicators import liquidity
from backtest.engine import forward_return, event_study

np.random.seed(11)

# ---------------------------------------------------------------------------
# 1) compute(): 유동성이 뚜렷하게 다른 두 구간을 인위적으로 만든다.
# ---------------------------------------------------------------------------
n = 400
dates = pd.date_range("2023-01-01", periods=n, freq="D")

# 전반부: 거래량 크고 가격 변동 작음(유동적) / 후반부: 거래량 작고 가격 변동 큼(비유동적)
close = np.empty(n)
close[0] = 100.0
ret = np.empty(n)
ret[0] = 0.0
for i in range(1, n):
    if i < n // 2:
        ret[i] = np.random.normal(0, 0.002)
    else:
        ret[i] = np.random.normal(0, 0.03)
    close[i] = close[i - 1] * (1 + ret[i])

volume = np.where(np.arange(n) < n // 2,
                   np.random.uniform(50000, 100000, n),
                   np.random.uniform(100, 500, n))

df = pd.DataFrame({
    "open": close, "high": close * 1.01, "low": close * 0.99,
    "close": close, "volume": volume,
}, index=dates)

liq_df = liquidity.compute(df, window=20, percentile_window=100)

assert {"dollar_volume", "daily_illiq", "amihud", "amihud_percentile"} <= set(liq_df.columns)

valid_pct = liq_df["amihud_percentile"].dropna()
assert not valid_pct.empty
assert valid_pct.between(0, 1).all(), "amihud_percentile은 0~1 범위여야 함"

late_amihud = liq_df["amihud"].iloc[-30:].mean()
early_amihud = liq_df["amihud"].iloc[100:130].mean()
assert late_amihud > early_amihud, \
    "거래대금 작고 변동성 큰 구간(후반)이 amihud(비유동성)가 더 커야 함"

# percentile_window=100이라 전환 직후(200봉 근방)는 trailing window 안에 유동적
# 구간과 비유동적 구간이 섞여 있어, 이 시점의 amihud_percentile이 전환 전(순수
# 유동적 구간, 200봉 이전)보다 뚜렷하게 높아야 한다.
before_switch_pct = liq_df["amihud_percentile"].iloc[150:180].mean()
just_after_switch_pct = liq_df["amihud_percentile"].iloc[220:250].mean()
assert just_after_switch_pct > before_switch_pct, \
    "비유동 구간 진입 직후 amihud_percentile이 그 전보다 높아야 함"

print(f"[OK] compute(): early_amihud={early_amihud:.3e} < late_amihud={late_amihud:.3e}, "
      f"before_switch_pct={before_switch_pct:.2f} < just_after_switch_pct={just_after_switch_pct:.2f}")

# ---------------------------------------------------------------------------
# 2) apply_filter(): 이벤트 컬럼을 만들어 필터가 실제로 마스킹하는지 확인.
# ---------------------------------------------------------------------------
fake_event = pd.Series(True, index=dates)  # 매일 이벤트가 뜬다고 가정
dfs_by_indicator = {"dummy": pd.DataFrame({"dummy_event": fake_event}, index=dates)}
signal_specs = [("dummy", "dummy_event", 1, "dummy_label")]

filtered = liquidity.apply_filter(dfs_by_indicator, liq_df, signal_specs, threshold=0.9)
out_col = filtered["dummy"]["dummy_event"]

illiquid_mask = liq_df["amihud_percentile"] > 0.9
assert (out_col[illiquid_mask.fillna(False)] == False).all(), \
    "amihud_percentile > threshold인 시점은 이벤트가 제외돼야 함"

nan_mask = liq_df["amihud_percentile"].isna()
assert nan_mask.any(), "테스트 구성상 초반에 NaN 구간이 있어야 함"
assert (out_col[nan_mask] == True).all(), "amihud_percentile이 NaN인 구간은 통과(True)해야 함"

liquid_mask = liq_df["amihud_percentile"] <= 0.9
assert (out_col[liquid_mask.fillna(False)] == True).all(), \
    "amihud_percentile <= threshold인 시점은 이벤트가 유지돼야 함"

print(f"[OK] apply_filter(): 비유동 구간 제외 {int((~out_col[illiquid_mask.fillna(False)]).sum() == illiquid_mask.fillna(False).sum())}, "
      f"NaN 구간 통과, 유동 구간 유지 모두 확인")

# ---------------------------------------------------------------------------
# 3) estimate_slippage_bps(): 선형 매핑 확인.
# ---------------------------------------------------------------------------
synth_liq = pd.DataFrame({"amihud_percentile": [0.0, 0.5, 1.0, np.nan]})
slip = liquidity.estimate_slippage_bps(synth_liq, base_bps=5.0, max_extra_bps=50.0)
assert np.isclose(slip.iloc[0], 5.0)
assert np.isclose(slip.iloc[1], 30.0)
assert np.isclose(slip.iloc[2], 55.0)
assert np.isclose(slip.iloc[3], 5.0), "NaN은 percentile=0으로 간주해 base_bps만 적용"
print(f"[OK] estimate_slippage_bps(): pct=0->{slip.iloc[0]}bp, pct=0.5->{slip.iloc[1]}bp, "
      f"pct=1->{slip.iloc[2]}bp, NaN->{slip.iloc[3]}bp")

# ---------------------------------------------------------------------------
# 4) forward_return()/event_study()가 동적(pd.Series) 슬리피지를 정확히 반영하는지.
# ---------------------------------------------------------------------------
small_dates = pd.date_range("2023-01-01", periods=10, freq="D")
small_close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109.0], index=small_dates)
small_df = pd.DataFrame({"close": small_close})

flat_slip = 10.0
r_flat = forward_return(small_df, horizon=2, slippage_bps=flat_slip)

dyn_slip = pd.Series(10.0, index=small_dates)
r_dyn_equal = forward_return(small_df, horizon=2, slippage_bps=dyn_slip)
assert np.allclose(r_flat.dropna(), r_dyn_equal.dropna()), \
    "모든 봉에서 동일한 슬리피지의 Series는 스칼라와 결과가 같아야 함"

dyn_slip_varying = pd.Series(np.linspace(0, 100, len(small_dates)), index=small_dates)
r_dyn_varying = forward_return(small_df, horizon=2, slippage_bps=dyn_slip_varying)
r_zero_slip = forward_return(small_df, horizon=2, slippage_bps=0.0)

diff = (r_zero_slip - r_dyn_varying).dropna()
expected_diff = (dyn_slip_varying / 10000.0).reindex(diff.index)
assert np.allclose(diff, expected_diff), \
    "봉별 슬리피지 차이가 forward_return 차이에 그대로 반영돼야 함"
print("[OK] forward_return(): pd.Series 슬리피지가 스칼라와 동등(동일값)하고, "
      "봉별로 다른 값일 때도 정확히 반영됨")

# event_study()도 동적 슬리피지를 받아 에러 없이 동작하는지 확인(회귀 방지).
event_col = pd.Series([True, False, True, False, False, True, False, False, False, False],
                       index=small_dates)
full_df = pd.DataFrame({
    "open": small_close.values, "high": small_close.values * 1.01,
    "low": small_close.values * 0.99, "close": small_close.values,
    "event": event_col.values,
}, index=small_dates)
res = event_study(full_df, event_col="event", horizons=(2,), max_curve_horizon=2,
                   direction=1, slippage_bps=dyn_slip_varying)
assert res.n_events == int(event_col.sum())
print(f"[OK] event_study(): pd.Series 슬리피지로도 정상 실행 (n_events={res.n_events})")

print("\nSMOKE TEST (indicators.liquidity: Amihud(2002) 비유동성 필터/동적 슬리피지) OK")
