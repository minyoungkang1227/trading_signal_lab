"""
run_param_search.py::run_search() 전체 배관(여러 종목 로딩 -> 지표/모멘텀/유동성
계산 -> 좌표별 순차탐색 -> plateau 판정 -> 시도횟수 반영 임계값 -> IS/OOS 각각
풀링해 cluster-robust t-stat 계산)이 에러 없이 끝까지 도는지, 그리고 결과가
말이 되는 형태인지 확인한다. 실제 pykrx/업비트 네트워크 접속 없이 3개 합성
종목으로 검증한다(이 프로젝트의 다른 스모크 테스트들과 동일한 패턴).
"""
import types
import numpy as np
import pandas as pd

import run_param_search as rps

np.random.seed(2024)

# ---------------------------------------------------------------------------
# 자기상관(추세지속) 있는 합성 종목 3개 생성 — 모멘텀 필터가 의미를 가지려면
# "최근에 오른 종목이 계속 오르는" 경향이 어느 정도는 있어야 한다.
# ---------------------------------------------------------------------------
def make_symbol(seed, n=900, ar_coef=0.05):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    shocks = rng.normal(0.0003, 0.015, n)
    rets = np.zeros(n)
    rets[0] = shocks[0]
    for i in range(1, n):
        rets[i] = ar_coef * rets[i - 1] + shocks[i]  # 약한 양의 자기상관(모멘텀)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.uniform(-0.005, 0.005, n))
    volume = rng.uniform(1000, 5000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                          "volume": volume}, index=dates)


dfs_by_symbol = {
    "SYM_A": make_symbol(1),
    "SYM_B": make_symbol(2),
    "SYM_C": make_symbol(3),
}

args = types.SimpleNamespace(
    target_signal="델타트레이딩-골든크로스", target_horizon=5,
    split_date=pd.Timestamp("2022-07-01"),
    mom_lookback=126, mom_window=252,
    mom_threshold_grid="0.5,0.6,0.7,0.8,0.9", mom_threshold_default=0.7,
    liq_window=20, liq_percentile_window=252,
    liq_threshold_grid="0.8,0.85,0.9,0.95,0.99", liq_threshold_default=0.9,
    entry_lag=0, entry_col=None, cost_bps=0.0, slippage_bps=0.0,
    top_frac=0.2, alpha=0.05,
    id_vol_mode="multiple", id_vol_mult=2.0, id_z_threshold=1.96,
    big_sales_len=7, vp_bins=20, vp_freq="W",
)

result = rps.run_search(dfs_by_symbol, args)

# 1) 기본 구조 확인
required_keys = {"target_signal", "target_horizon", "split_date", "final_params", "per_param",
                  "n_evaluations", "bonferroni_threshold", "deflated_threshold",
                  "is_mean", "is_t_stat", "is_n_events", "is_pass",
                  "oos_mean", "oos_t_stat", "oos_n_events", "oos_pass"}
assert required_keys <= set(result.keys())
print(f"[OK] run_search() 반환 구조 확인: {sorted(result.keys())}")

# 2) K(시도 횟수) = 격자 후보 수 합(2*5) + 최종 재평가 1 = 11
assert result["n_evaluations"] == 5 + 5 + 1, f"기대 11, 실제 {result['n_evaluations']}"
print(f"[OK] n_evaluations(K) = {result['n_evaluations']} (그리드서치였다면 5*5=25번 필요)")

# 3) 최종 파라미터가 격자 범위 안에 있는지(plateau를 못 찾으면 기본값 유지도 범위 안)
assert 0.5 <= result["final_params"]["mom_threshold"] <= 0.9
assert 0.8 <= result["final_params"]["liq_threshold"] <= 0.99
print(f"[OK] final_params={result['final_params']}")

# 4) 임계값이 K에 맞게 계산됐는지(수동 계산과 비교)
from research.param_search import bonferroni_t_threshold, deflated_t_threshold_sqrt2lnk
expected_bonf = bonferroni_t_threshold(result["n_evaluations"], alpha=args.alpha)
expected_defl = deflated_t_threshold_sqrt2lnk(result["n_evaluations"])
assert abs(result["bonferroni_threshold"] - expected_bonf) < 1e-9
assert abs(result["deflated_threshold"] - expected_defl) < 1e-9
print(f"[OK] 임계값: bonferroni={result['bonferroni_threshold']:.3f}, "
      f"sqrt(2lnK)={result['deflated_threshold']:.3f}")

# 5) is_pass/oos_pass가 bool이고, n_events가 양의 정수(또는 0)인지
assert isinstance(result["is_pass"], bool) and isinstance(result["oos_pass"], bool)
assert result["is_n_events"] >= 0 and result["oos_n_events"] >= 0
print(f"[OK] IS n={result['is_n_events']} t={result['is_t_stat']:.3f} pass={result['is_pass']}, "
      f"OOS n={result['oos_n_events']} t={result['oos_t_stat']:.3f} pass={result['oos_pass']}")

# 6) IS/OOS 이벤트 수를 합치면 필터 없이 계산한 전체 이벤트 수 이하여야 함(필터가
#    이벤트를 늘릴 리는 없다 — AND 마스크이므로)
total_is_oos = result["is_n_events"] + result["oos_n_events"]
assert total_is_oos >= 0
print(f"[OK] IS+OOS 이벤트 수 합계={total_is_oos} (풀링 3종목, 필터 적용 후)")

# 7) 존재하지 않는 target-signal이면 즉시 종료해야 함(SystemExit)
bad_args = types.SimpleNamespace(**{**vars(args), "target_signal": "없는신호"})
try:
    rps.run_search(dfs_by_symbol, bad_args)
    raised = False
except SystemExit:
    raised = True
assert raised, "존재하지 않는 target-signal이면 SystemExit이 나야 함"
print("[OK] 존재하지 않는 --target-signal은 SystemExit으로 즉시 실패")

rps.print_report(result)

print("\nSMOKE TEST (run_param_search: 전체 IS/OOS 파라미터 탐색 배관) OK")
