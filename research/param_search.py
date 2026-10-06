"""
IS(In-Sample)에서 파라미터를 정하고 OOS(Out-of-Sample)에서 검증하는 절차를
"제대로" 하려고 하면 세 가지 문제가 얽힌다.

  1. 파라미터가 여러 개라 조합을 다 시도하면(그리드서치) 금방 폭발한다.
  2. IS에서 t-stat이 제일 높은 "단 하나의" 조합을 고르는 순간, 그 자체가 또
     하나의 다중검정(수백 개 조합 중 우연히 좋아 보인 것을 골랐을 위험)이 된다.
  3. 종목 하나로만 보면 이벤트 수(n_events)가 몇십 개 수준이라 표본이 작다.

이 모듈은 이 세 문제를 각각 겨냥한 도구를 모아둔다.

  - sequential_anchor_search() : 그리드서치(C^P) 대신 좌표별 순차탐색(P*C)으로
    1번(조합 폭발)을 줄인다.
  - detect_plateau()           : IS에서 "최고점 1개"가 아니라 "성과가 유지되는
    평탄 구간"을 찾아, 뾰족한 산(peak, 과최적화 의심)과 완만한 언덕(plateau,
    강인한 파라미터)을 구분한다. 2번을 완전히 없애진 못하지만 줄여준다.
  - bonferroni_t_threshold() / deflated_t_threshold_sqrt2lnk()
                                : 그래도 여러 번 시도한 사실 자체를 유의성
    기준에 반영한다(2번에 대한 마지막 방어선). Bailey, Borwein, López de
    Prado & Zhu(2014)와 Harvey, Liu & Zhu(2016)가 쓰는 두 가지 방식을 각각
    구현했다 — 하나만 정답이라고 믿지 말고 참고용으로 같이 본다.
  - pool_event_returns() / cluster_robust_t_stat() (cluster_robust_t_stat은
    backtest.engine에 있음)
                                : 3번(표본 부족)을 여러 종목 풀링으로 늘리되,
    풀링이 만드는 "같은 날 여러 종목이 같이 움직이는" 새로운 상관 문제를
    날짜 클러스터링으로 보정한다.

주의: 이 모듈의 어떤 함수도 "이 파라미터가 최적이다"라는 강한 주장을 하지
않는다. sequential_anchor_search()가 IS에서 골라주는 값은 "IS 안에서 안전하게
좁힌 후보"일 뿐이고, 그 값이 실제로 쓸 만한지는 반드시 이 탐색 과정이 전혀
들여다보지 않은 OOS 구간에서 별도로 확인해야 한다(run_param_search.py가 이
순서를 강제한다).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.engine import p_value_from_t


# ---------------------------------------------------------------------------
# 시도 횟수(K)를 반영한 유의성 임계값
# ---------------------------------------------------------------------------
def bonferroni_t_threshold(k_trials: int, alpha: float = 0.05) -> float:
    """
    K번의 검정을 시도했다는 사실 자체를 반영해, 전체 1종 오류율을 alpha 이하로
    묶기 위한 본페로니 보정 임계 |t-stat|. 목표 p-value(alpha/K)를 만족하는
    t값을 이분법으로 역산한다(scipy 없이 backtest.engine.p_value_from_t만으로
    — 이 프로젝트 전체의 무의존성 설계와 동일).

    K개 시행이 서로 독립이라는 가정 아래에서는 정확한 보정이고(Bonferroni는
    독립이 아니어도 항상 보수적인 쪽으로만 어긋난다 — union bound라 실제
    필요한 것보다 더 엄격한 기준을 줄지언정, 너무 느슨해지진 않는다), K가
    작을 때도 sqrt(2*ln(K)) 근사보다 오차가 적다.
    """
    if k_trials < 1:
        raise ValueError("k_trials >= 1 이어야 합니다")
    if k_trials == 1:
        # 시행이 1번뿐이면 다중검정 보정이 필요 없다 — 목표 alpha 자체의
        # 양측검정 임계값을 그대로 반환.
        target_p = alpha
    else:
        target_p = alpha / k_trials

    lo, hi = 0.0, 40.0  # |t|=40이면 p-value가 사실상 0이라 이 범위면 충분
    for _ in range(100):
        mid = (lo + hi) / 2
        if p_value_from_t(mid) > target_p:
            lo = mid
        else:
            hi = mid
    return hi


def deflated_t_threshold_sqrt2lnk(k_trials: int) -> float:
    """
    Bailey, Borwein, López de Prado & Zhu(2014)/Harvey, Liu & Zhu(2016)류가
    쓰는 "K개의 독립 표준정규 시행 중 최댓값의 기댓값" 근사: sqrt(2*ln(K)).

    bonferroni_t_threshold()보다 유도가 단순하고 관행적으로 많이 인용되지만,
    K가 작을 때(대략 50 미만) 근사오차가 꽤 커질 수 있다는 점에 유의 — 이
    구간에서는 bonferroni_t_threshold()가 더 정확하다. run_param_search.py는
    둘 다 계산해서 나란히 보여준다(하나의 숫자를 정답처럼 내세우지 않기 위해).
    """
    if k_trials < 2:
        return 0.0
    return math.sqrt(2.0 * math.log(k_trials))


# ---------------------------------------------------------------------------
# Peak(과최적화 의심) vs Plateau(강인한 구간) 판정
# ---------------------------------------------------------------------------
@dataclass
class PlateauResult:
    is_plateau: bool
    robust_value: float | None
    plateau_candidates: list = field(default_factory=list)
    plateau_metrics: list = field(default_factory=list)
    reason: str = ""


def detect_plateau(candidates: list[float], metrics: list[float],
                    top_frac: float = 0.2, max_gap: int = 1) -> PlateauResult:
    """
    candidates: 파라미터 후보값. 오름차순 여부는 상관없다(내부에서 정렬한다) —
        단, 값 자체에 순서가 있는 연속형/순서형 파라미터에만 이 함수가 의미가
        있다("multiple" vs "zscore" 같은 범주형 파라미터에는 쓰지 말 것 —
        "인접"이라는 개념 자체가 성립하지 않는다).
    metrics: 각 후보를 objective_fn으로 평가한 성과지표(보통 t-stat). NaN은
        "평가 불가"로 보고 판정에서 제외한다.
    top_frac: 상위 몇 %를 "성과 좋은 후보"로 볼지(기본 상위 20%).
    max_gap: 상위 후보들의 격자 위치(순위가 아니라 정렬된 격자에서의 인덱스)
        사이에 이 정도 간격까지는 "이어져 있다"고 봐준다(기본 1 = 바로 옆
        칸 하나 정도 비어도 이어진 것으로 간주 — 격자가 촘촘하지 않을 때의
        여유).

    반환: PlateauResult
      is_plateau=True  : 상위 후보들이 격자상에서 하나의 이어진 구간을 이룸
          (peak가 아니라 plateau). 그 구간의 중앙값을 robust_value로 준다.
      is_plateau=False : 상위 후보들이 여러 구간으로 흩어져 있거나(전형적인
          과최적화 스파이크 패턴), 유효 후보가 너무 적어 판정이 불가능함.
          robust_value=None — 호출부(sequential_anchor_search)는 이 파라미터를
          기본값으로 되돌리는 걸 권장한다.

    주의: 이 함수가 "plateau"라고 판정해도 그 자체로 편향이 사라지는 건
    아니다(top_frac 후보들도 결국 IS 성과 기준으로 골랐으므로). 최종 확인은
    반드시 이 판정에 쓰이지 않은 OOS 데이터로 별도 진행해야 한다.
    """
    cand = np.asarray(candidates, dtype=float)
    met = np.asarray(metrics, dtype=float)
    valid = ~np.isnan(met)
    if valid.sum() < 3:
        return PlateauResult(False, None, [], [],
                              "유효 후보가 3개 미만이라 plateau 판정 불가")

    cand_v = cand[valid]
    met_v = met[valid]
    order = np.argsort(cand_v)
    cand_sorted = cand_v[order]
    met_sorted = met_v[order]

    n = len(cand_sorted)
    top_n = max(1, int(math.ceil(n * top_frac)))
    top_positions = sorted(np.argsort(met_sorted)[::-1][:top_n].tolist())

    groups = [[top_positions[0]]]
    for p in top_positions[1:]:
        if p - groups[-1][-1] <= max_gap + 1:
            groups[-1].append(p)
        else:
            groups.append([p])

    largest_group = max(groups, key=len)
    is_plateau = len(groups) == 1 and len(largest_group) >= max(2, top_n // 2)

    plateau_cands = cand_sorted[top_positions].tolist()
    plateau_mets = met_sorted[top_positions].tolist()

    if not is_plateau:
        return PlateauResult(
            False, None, plateau_cands, plateau_mets,
            f"상위 {top_n}개 후보가 {len(groups)}개 구간으로 흩어져 있음 "
            f"(peak 패턴 — 과최적화 의심, 기본값 유지 권장)")

    block = cand_sorted[largest_group]
    robust_value = float(np.median(block))
    return PlateauResult(
        True, robust_value, plateau_cands, plateau_mets,
        f"상위 {top_n}개 후보가 이어진 구간[{block.min():.3g}~{block.max():.3g}]을 "
        f"이룸(plateau) — 중앙값 {robust_value:.3g} 채택")


# ---------------------------------------------------------------------------
# 좌표별 순차 탐색 (Anchor Search) — 그리드서치(C^P) 대신 P*C
# ---------------------------------------------------------------------------
def sequential_anchor_search(param_grid: dict[str, list[float]], defaults: dict,
                              objective_fn, top_frac: float = 0.2, max_gap: int = 1) -> dict:
    """
    좌표별(coordinate-wise) 순차 탐색. 파라미터 P개 x 후보 C개를 전부 조합하는
    그리드서치는 C^P번 평가해야 하지만, 여기서는 한 번에 파라미터 하나씩만
    바꿔가며(다른 건 지금까지 확정된 값 또는 기본값으로 고정) 최선의 값을
    찾으므로 P*C번이면 끝난다.

    대가: 파라미터 간 상호작용(교호작용 — 예: liq_threshold가 어떤 값일 때
    mom_threshold의 최적치가 달라지는 경우)은 놓칠 수 있다. 지금처럼 표본이
    작은 상황에서는 "조합 폭발 + 그로 인한 과최적화" 리스크가 "상호작용을
    못 잡는" 리스크보다 크다고 보고 택한 절충이다.

    param_grid: {"mom_threshold": [0.5, 0.6, 0.7, 0.8, 0.9], ...}. 순서형
        파라미터만 넣을 것(범주형은 detect_plateau가 의미가 없다).
    defaults: 각 파라미터의 시작값(보통 논문 근거로 고른 기존 기본값). 탐색
        중간에 다른 파라미터를 고정할 때도 이 값이 쓰인다.
    objective_fn(params: dict) -> float: 주어진 전체 파라미터 조합(다른 고정
        파라미터 포함)에 대해 성과지표(보통 pooled cluster-robust t-stat)를
        반환하는 콜백. 평가 불가면 NaN을 반환하면 된다.
    top_frac, max_gap: detect_plateau()에 그대로 전달.

    반환: dict(
        final_params : {파라미터명: 확정값}. plateau를 못 찾은 파라미터는
            defaults의 값 그대로 유지된다(탐색하지 않은 것과 동일하게 취급 —
            애매한 상황에서 억지로 값을 바꾸지 않는다).
        per_param    : {파라미터명: PlateauResult} — 각 파라미터의 판정 근거.
        n_evaluations: 이 탐색에서 objective_fn을 호출한 총 횟수. 이게 바로
            deflated_t_threshold_*()에 넣을 K다(그리드서치였다면 훨씬 큰 수가
            나왔을 것 — 순차탐색이 K 자체를 줄여준다는 게 핵심 이득 중 하나).
    )
    """
    current = dict(defaults)
    per_param: dict[str, PlateauResult] = {}
    n_evaluations = 0

    for param_name, candidates in param_grid.items():
        metrics = []
        for val in candidates:
            trial_params = dict(current)
            trial_params[param_name] = val
            metrics.append(objective_fn(trial_params))
            n_evaluations += 1

        plateau = detect_plateau(candidates, metrics, top_frac=top_frac, max_gap=max_gap)
        per_param[param_name] = plateau
        if plateau.is_plateau:
            current[param_name] = plateau.robust_value
        # else: 기본값 유지. peak로 판정된 파라미터를 억지로 "최고 성과 후보"로
        # 바꾸면 바로 과최적화로 이어지므로, 이 파라미터는 탐색 안 한 셈 친다.

    return dict(final_params=current, per_param=per_param, n_evaluations=n_evaluations)


# ---------------------------------------------------------------------------
# 횡단면 풀링 — 여러 종목의 이벤트 수익률을 하나의 표본으로 합친다
# ---------------------------------------------------------------------------
def pool_event_returns(per_symbol: dict[str, pd.Series]) -> tuple[np.ndarray, np.ndarray]:
    """
    per_symbol: {"005930": event_returns_series, "000660": ..., ...} — 각각
        backtest.engine.event_returns()의 반환값(이벤트 날짜를 인덱스로 하는
        pd.Series)이어야 한다.

    반환: (values, cluster_ids) — 둘 다 같은 길이의 numpy array. cluster_ids는
    이벤트 날짜를 정수(date.toordinal())로 바꾼 값이다 — 같은 날짜에 여러
    종목의 이벤트가 몰리면 자동으로 같은 클러스터가 되어, 이어서
    backtest.engine.cluster_robust_t_stat()에 넘기면 그 횡단면 상관을 감안한
    t-stat을 계산할 수 있다. 빈 입력이면 빈 배열 두 개를 반환한다.
    """
    all_vals, all_dates = [], []
    for series in per_symbol.values():
        if series is None or len(series) == 0:
            continue
        all_vals.append(np.asarray(series.values, dtype=float))
        all_dates.append(np.array([pd.Timestamp(d).toordinal() for d in series.index]))

    if not all_vals:
        return np.array([]), np.array([])
    return np.concatenate(all_vals), np.concatenate(all_dates)


# ---------------------------------------------------------------------------
# IS/OOS 분할 — "고정 규칙"의 시간순 재현성 검증
# ---------------------------------------------------------------------------
def split_is_oos(event_returns_series: pd.Series, split_date) -> tuple[pd.Series, pd.Series]:
    """
    이벤트 날짜 기준으로 IS(<= split_date)와 OOS(> split_date)로 나눈다.

    지표(momentum/liquidity 롤링윈도우 포함)는 항상 전체 기간 데이터로 미리
    계산해두고, 이 함수는 "이미 계산된 이벤트 수익률"을 날짜로만 자르는
    역할만 한다 — 그래야 OOS 구간 초반이 롤링윈도우 워밍업 때문에 표본을
    잃는 일이 없다(지표 계산과 IS/OOS 분할을 분리하는 게 핵심 설계 포인트).
    """
    split_date = pd.Timestamp(split_date)
    is_part = event_returns_series[event_returns_series.index <= split_date]
    oos_part = event_returns_series[event_returns_series.index > split_date]
    return is_part, oos_part
