"""
backtest/engine.py의 다중 검정 보정(Benjamini-Hochberg) 로직 검증.
p_value_from_t()가 정규분포 근사로 올바른 p-value를 내는지, benjamini_hochberg()가
교과서적 BH 절차를 정확히 재현하는지, compare_all()의 요약표에 p_value·
significant_bh 컬럼이 잘 붙는지 세 단계로 확인한다.
"""
import math
import numpy as np
import pandas as pd
from backtest.engine import p_value_from_t, benjamini_hochberg, compare_all
from indicators import institutional_displacement, delta_trading, big_sales, ultimate_rsi, vp_box

# 1) p_value_from_t: 잘 알려진 기준값과 대조 (양측검정, 표준정규분포 근사)
#    t=1.96 -> p≈0.05, t=0 -> p=1.0, NaN -> NaN
p_196 = p_value_from_t(1.959963984540054)  # 95% 임계값 정확히
assert abs(p_196 - 0.05) < 1e-6, p_196
assert abs(p_value_from_t(0.0) - 1.0) < 1e-9
assert math.isnan(p_value_from_t(float("nan")))
assert abs(p_value_from_t(-1.959963984540054) - 0.05) < 1e-6, "부호 무관 양측검정이어야 함"
print(f"[OK] p_value_from_t: t=1.96 -> p={p_196:.4f}(기대 0.05), 부호 무관, NaN 처리 확인")

# 2) benjamini_hochberg: 교과서 예제(위키피디아 BH 절차 예시와 동일한 손 계산)
#    p = [0.005, 0.011, 0.019, 0.022, 0.driver] 같은 임의 값 대신, 직접 설계한 값으로
#    "몇 번째까지 유의한지"를 손으로 계산해 대조한다.
#    m=5, fdr=0.10 -> 임계값(i/m*fdr) = [0.02, 0.04, 0.06, 0.08, 0.10]
p_values = np.array([0.001, 0.03, 0.05, 0.07, 0.20])
#   순위(오름차순 그대로) 1~5, p_(i) <= threshold?
#   1: 0.001 <= 0.02 True
#   2: 0.03  <= 0.04 True
#   3: 0.05  <= 0.06 True
#   4: 0.07  <= 0.08 True
#   5: 0.20  <= 0.10 False
#   -> True인 것 중 가장 큰 순위 = 4 -> 순위 1~4까지 전부 유의, 5번째만 기각
expected = np.array([True, True, True, True, False])
result = benjamini_hochberg(p_values, fdr=0.10)
assert np.array_equal(result, expected), f"BH 절차 결과 불일치: {result} vs {expected}"
print(f"[OK] benjamini_hochberg 교과서 예제 재현: {result}")

# 순서가 섞여 있어도(정렬 없이 넣어도) 같은 항목이 같은 판정을 받는지 확인
shuffled_idx = [4, 0, 3, 1, 2]
p_shuffled = p_values[shuffled_idx]
result_shuffled = benjamini_hochberg(p_shuffled, fdr=0.10)
expected_shuffled = expected[shuffled_idx]
assert np.array_equal(result_shuffled, expected_shuffled)
print("[OK] 입력 순서가 섞여도 각 항목의 판정은 동일 (순위만 내부적으로 재계산)")

# NaN 포함 시 그 항목은 항상 False, 나머지는 유효한 것들끼리만 다시 랭킹
p_with_nan = np.array([0.001, np.nan, 0.05])
result_nan = benjamini_hochberg(p_with_nan, fdr=0.10)
assert result_nan[1] == False
print(f"[OK] NaN 포함 시 해당 항목은 항상 False: {result_nan}")

# 3) compare_all() 통합 — 요약표에 p_value/significant_bh 컬럼이 실제로 붙는지
np.random.seed(1)
n = 400
dates = pd.date_range("2023-01-01", periods=n, freq="D")
close = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, n)))
high = close * (1 + np.random.uniform(0, 0.01, n))
low = close * (1 - np.random.uniform(0, 0.01, n))
open_ = close * (1 + np.random.uniform(-0.005, 0.005, n))
volume = np.random.uniform(1000, 5000, n)
df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates)

dfs_by_indicator = {
    "institutional_displacement": institutional_displacement.compute(df),
    "delta_trading": delta_trading.compute(df),
    "big_sales": big_sales.compute(df),
    "ultimate_rsi": ultimate_rsi.compute(df),
    "vp_box": vp_box.compute(df),
}
summary, results = compare_all(dfs_by_indicator, horizons=(1, 5, 10), fdr=0.10)
assert "p_value" in summary.columns and "significant_bh" in summary.columns
# t_stat이 NaN인 행은 p_value도 NaN, significant_bh도 False여야 함
nan_rows = summary[summary["t_stat"].isna()]
if len(nan_rows) > 0:
    assert nan_rows["p_value"].isna().all()
    assert (~nan_rows["significant_bh"]).all()
# significant_bh가 True인 행은 반드시 |t_stat|이 낮은 유의수준(예: p<=fdr) 근방이어야 함(느슨한 정합성 체크)
sig_rows = summary[summary["significant_bh"]]
if len(sig_rows) > 0:
    assert (sig_rows["p_value"] <= 0.10).all(), "significant_bh=True인데 p_value가 fdr보다 큰 행이 있음"
print(f"\ncompare_all() 요약표 shape={summary.shape}, significant_bh=True인 행 수={summary['significant_bh'].sum()}")
print("[OK] compare_all()에 p_value/significant_bh 컬럼이 정상적으로 붙고, 논리적 정합성도 확인됨")

print("\nSMOKE TEST (Benjamini-Hochberg) OK")
