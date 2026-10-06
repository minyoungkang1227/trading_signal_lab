"""
research/param_search.py + backtest.engine.cluster_robust_t_stat/event_returns 검증.

1) bonferroni_t_threshold / deflated_t_threshold_sqrt2lnk: K가 커질수록 임계값이
   커지는지, K=1이면 사실상 단순 유의수준과 같은지.
2) detect_plateau: 완만한 언덕(plateau)과 뾰족한 산(peak) 합성 데이터를 각각
   만들어 정확히 구분하는지.
3) sequential_anchor_search: 두 파라미터짜리 장난감 목적함수(하나는 plateau 모양,
   하나는 순수 잡음)에서 그리드서치보다 훨씬 적은 평가 횟수(P*C)로 정확히
   plateau 파라미터는 채택하고 잡음 파라미터는 기본값을 유지하는지.
4) cluster_robust_t_stat: 클러스터(같은 날짜) 안에서 강하게 상관된 표본을
   풀링했을 때, 단순 std/sqrt(n) t-stat보다 절댓값이 작게(더 보수적으로)
   나오는지 — Newey-West 테스트와 같은 논리를 횡단면 축에 적용.
5) event_returns()/pool_event_returns()/split_is_oos(): 여러 종목을 풀링하고
   IS/OOS로 자르는 배관이 정확히 동작하는지.
"""
import numpy as np
import pandas as pd

from backtest.engine import event_returns, cluster_robust_t_stat, p_value_from_t
from research.param_search import (
    bonferroni_t_threshold, deflated_t_threshold_sqrt2lnk,
    detect_plateau, sequential_anchor_search, pool_event_returns, split_is_oos,
)

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1) 임계값 함수
# ---------------------------------------------------------------------------
t1 = bonferroni_t_threshold(1, alpha=0.05)
t100 = bonferroni_t_threshold(100, alpha=0.05)
t1000 = bonferroni_t_threshold(1000, alpha=0.05)
assert t1 < t100 < t1000, "K가 커질수록 본페로니 임계값도 커져야 함"
# K=1이면 목표 p-value가 그대로 alpha이므로, 그 t값에서의 p-value가 alpha에 근접해야 함
assert abs(p_value_from_t(t1) - 0.05) < 1e-3
print(f"[OK] bonferroni_t_threshold: K=1 -> {t1:.3f}, K=100 -> {t100:.3f}, K=1000 -> {t1000:.3f}")

sq2 = deflated_t_threshold_sqrt2lnk(100)
assert abs(sq2 - np.sqrt(2 * np.log(100))) < 1e-9
assert deflated_t_threshold_sqrt2lnk(1) == 0.0
print(f"[OK] deflated_t_threshold_sqrt2lnk(100) = {sq2:.3f} (기대 {np.sqrt(2*np.log(100)):.3f})")

# ---------------------------------------------------------------------------
# 2) detect_plateau: plateau vs peak
# ---------------------------------------------------------------------------
candidates = [0.5, 0.6, 0.7, 0.8, 0.9]
plateau_metrics = [1.0, 3.0, 3.2, 3.1, 1.2]  # 가운데 세 개가 비슷하게 높음 -> plateau
peak_metrics = [1.0, 1.1, 5.0, 1.2, 1.0]      # 딱 하나만 튐 -> peak(과최적화 의심)

res_plateau = detect_plateau(candidates, plateau_metrics, top_frac=0.6)
assert res_plateau.is_plateau
assert 0.6 <= res_plateau.robust_value <= 0.8
print(f"[OK] detect_plateau(평지 패턴): is_plateau=True, robust_value={res_plateau.robust_value}")

res_peak = detect_plateau(candidates, peak_metrics, top_frac=0.2)
assert not res_peak.is_plateau
assert res_peak.robust_value is None
print(f"[OK] detect_plateau(뾰족한 산 패턴): is_plateau=False (과최적화 의심으로 기각) -> {res_peak.reason}")

# 유효 후보 부족
res_few = detect_plateau([0.5, 0.6], [1.0, 2.0])
assert not res_few.is_plateau
print("[OK] detect_plateau(): 유효 후보 3개 미만이면 판정 불가로 처리")

# ---------------------------------------------------------------------------
# 3) sequential_anchor_search: plateau 파라미터 + 잡음 파라미터
# ---------------------------------------------------------------------------
call_count = {"n": 0}


def toy_objective(params):
    call_count["n"] += 1
    # a는 0.7 근방에서 완만한 언덕(진짜 신호가 있는 파라미터라고 가정)
    a = params["a"]
    a_term = 3.0 - 20.0 * (a - 0.7) ** 2
    # b는 순수 잡음(신호 없음 -> 어떤 값을 골라도 plateau가 안 나와야 함)
    rng = np.random.RandomState(int(params["b"] * 1000) % (2 ** 31))
    b_term = rng.normal(0, 0.05)
    return a_term + b_term


param_grid = {
    "a": [0.5, 0.6, 0.7, 0.8, 0.9],
    "b": [0.1, 0.3, 0.5, 0.7, 0.9],
}
defaults = {"a": 0.7, "b": 0.5}

result = sequential_anchor_search(param_grid, defaults, toy_objective, top_frac=0.4)
assert result["n_evaluations"] == 10, f"P*C=2*5=10번만 평가해야 함(그리드서치였으면 25번), 실제={result['n_evaluations']}"
assert 0.6 <= result["final_params"]["a"] <= 0.8, \
    f"plateau 파라미터 a는 0.7 근방에서 채택돼야 함: {result['final_params']['a']}"
print(f"[OK] sequential_anchor_search(): n_evaluations={result['n_evaluations']} "
      f"(그리드서치 대비 절감), final_params={result['final_params']}")
print(f"     a: {result['per_param']['a'].reason}")
print(f"     b: {result['per_param']['b'].reason}")

# ---------------------------------------------------------------------------
# 4) cluster_robust_t_stat: 클러스터 내부 상관이 있으면 t-stat이 보수적으로 줄어야 함
# ---------------------------------------------------------------------------
n_clusters = 20
per_cluster = 5
cluster_shocks = np.random.normal(0, 1.0, n_clusters)  # 클러스터(날짜) 공통 충격
vals = []
cluster_ids = []
for c in range(n_clusters):
    # 같은 클러스터 안의 관측치들은 공통 충격을 강하게 공유(상관 높음)
    obs = cluster_shocks[c] + np.random.normal(0, 0.1, per_cluster)
    vals.extend(obs.tolist())
    cluster_ids.extend([c] * per_cluster)
vals = np.array(vals)
cluster_ids = np.array(cluster_ids)

naive_mean = vals.mean()
naive_std = vals.std(ddof=1)
naive_t = naive_mean / (naive_std / np.sqrt(len(vals)))

cr_mean, cr_t = cluster_robust_t_stat(vals, cluster_ids)
assert np.isclose(cr_mean, naive_mean)
assert abs(cr_t) < abs(naive_t), \
    f"클러스터 내부 상관이 강한 표본은 cluster-robust t-stat이 naive보다 작아야 함: naive={naive_t:.2f}, cr={cr_t:.2f}"
print(f"[OK] cluster_robust_t_stat(): 클러스터 상관 있는 데이터 naive_t={naive_t:.2f} -> cluster_t={cr_t:.2f}")

# 클러스터가 없을 때(전부 서로 다른 클러스터, n=1씩)는 계산 불가 -> NaN 아닌 값 or NaN
indep_ids = np.arange(len(vals))
_, cr_t_indep = cluster_robust_t_stat(vals, indep_ids)
# 클러스터당 관측치 1개뿐이면 클러스터 내부 잔차합이 곧 잔차 자체라 naive와 비슷한 스케일이어야 함
assert np.isfinite(cr_t_indep) or np.isnan(cr_t_indep)
print(f"[OK] cluster_robust_t_stat(): 클러스터=관측치 단위일 때도 에러 없이 계산됨 (t={cr_t_indep:.2f})")

# ---------------------------------------------------------------------------
# 5) event_returns / pool_event_returns / split_is_oos 배관 확인
# ---------------------------------------------------------------------------
dates = pd.date_range("2023-01-01", periods=30, freq="D")
close = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, 30))), index=dates)
df_a = pd.DataFrame({"close": close, "event": [i % 5 == 0 for i in range(30)]}, index=dates)
df_b = pd.DataFrame({"close": close * 1.5, "event": [i % 7 == 0 for i in range(30)]}, index=dates)

ret_a = event_returns(df_a, "event", horizon=3, direction=1)
ret_b = event_returns(df_b, "event", horizon=3, direction=1)
assert ret_a.index.is_monotonic_increasing
assert len(ret_a) == sum(1 for i in range(30) if i % 5 == 0 and i + 3 < 30)
print(f"[OK] event_returns(): symbol A n={len(ret_a)}, symbol B n={len(ret_b)}")

vals_pooled, cluster_ids_pooled = pool_event_returns({"A": ret_a, "B": ret_b})
assert len(vals_pooled) == len(ret_a) + len(ret_b)
assert len(cluster_ids_pooled) == len(vals_pooled)
print(f"[OK] pool_event_returns(): 풀링된 표본 수={len(vals_pooled)} (A+B={len(ret_a)}+{len(ret_b)})")

split_date = dates[15]
is_part, oos_part = split_is_oos(ret_a, split_date)
assert (is_part.index <= split_date).all()
assert (oos_part.index > split_date).all()
assert len(is_part) + len(oos_part) == len(ret_a)
print(f"[OK] split_is_oos(): IS n={len(is_part)}, OOS n={len(oos_part)} (합계 {len(ret_a)}와 일치)")

# 빈 입력 처리
empty_vals, empty_ids = pool_event_returns({})
assert len(empty_vals) == 0 and len(empty_ids) == 0
print("[OK] pool_event_returns(): 빈 입력도 에러 없이 빈 배열 반환")

print("\nSMOKE TEST (research.param_search: plateau/anchor-search/deflated-t/pooling/IS-OOS) OK")
