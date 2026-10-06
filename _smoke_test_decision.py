"""
decision/rule_based.py의 규칙 분기를 전부 검증한다. judge_signal()은 순수 함수라
케이스별로 입력만 바꿔서 일곱 가지 분기(보류 3종 + 기각 3종 + 채택 1종)가 전부
의도한 판정을 내는지 확인하고, judge_indicator()는 event_study()·event_alpha()와
실제로 엮여서 끝까지 동작하는지 합성 데이터로 확인한다.

분기 자체를 테스트하는 값들은 t_threshold=2.0을 명시적으로 고정해서, 이후
Harvey/Liu/Zhu(2016) 권고를 반영해 t_threshold의 "기본값"을 3.0으로 올리더라도
이 테스트들이 영향받지 않게 했다(분기 로직 검증과 기본값 검증은 서로 다른
관심사이므로 분리). 기본값이 실제로 3.0인지는 별도 테스트로 확인한다.
"""
import numpy as np
import pandas as pd
from decision.rule_based import judge_signal, judge_indicator

T = 2.0  # 분기 로직 테스트에 쓸 고정 임계값(기본값 변경과 무관하게 유지)

# 1) 표본 부족 -> 보류
r = judge_signal(n_events=5, raw_mean_return=0.02, raw_t_stat=3.0,
                  alpha=0.01, alpha_t_stat=2.5, r_squared=0.5, t_threshold=T)
assert r["verdict"] == "보류", r

# 2) 원 신호 자체가 유의하지 않음 -> 기각
r = judge_signal(n_events=50, raw_mean_return=0.005, raw_t_stat=0.8,
                  alpha=0.001, alpha_t_stat=0.3, r_squared=0.1, t_threshold=T)
assert r["verdict"] == "기각", r

# 3) 알파 t-stat이 NaN(회귀 표본 부족) -> 보류
r = judge_signal(n_events=50, raw_mean_return=0.02, raw_t_stat=3.0,
                  alpha=float("nan"), alpha_t_stat=float("nan"), r_squared=float("nan"), t_threshold=T)
assert r["verdict"] == "보류", r

# 4) 원 신호는 유의하나 알파는 유의하지 않음(팩터로 설명됨) -> 기각
r = judge_signal(n_events=50, raw_mean_return=0.02, raw_t_stat=3.0,
                  alpha=0.001, alpha_t_stat=0.5, r_squared=0.6, t_threshold=T)
assert r["verdict"] == "기각", r

# 5) 원 수익률과 알파 부호가 다름 -> 보류
r = judge_signal(n_events=50, raw_mean_return=0.02, raw_t_stat=3.0,
                  alpha=-0.01, alpha_t_stat=-2.5, r_squared=0.4, t_threshold=T)
assert r["verdict"] == "보류", r

# 6) 전부 통과 -> 채택
r = judge_signal(n_events=50, raw_mean_return=0.02, raw_t_stat=3.0,
                  alpha=0.01, alpha_t_stat=2.5, r_squared=0.4, t_threshold=T)
assert r["verdict"] == "채택", r

# 7) bh_significant=False -> 개별 t-stat이 유의해도 기각(다중검정 보정 실패)
r = judge_signal(n_events=50, raw_mean_return=0.02, raw_t_stat=3.0,
                  alpha=0.01, alpha_t_stat=2.5, r_squared=0.4, t_threshold=T, bh_significant=False)
assert r["verdict"] == "기각", r
assert "다중검정" in r["reasons"][0] or "Benjamini" in r["reasons"][0], r["reasons"]

# 8) bh_significant=True -> 다른 조건 전부 통과 시 그대로 채택(회귀 없음 확인)
r = judge_signal(n_events=50, raw_mean_return=0.02, raw_t_stat=3.0,
                  alpha=0.01, alpha_t_stat=2.5, r_squared=0.4, t_threshold=T, bh_significant=True)
assert r["verdict"] == "채택", r

# 9) bh_significant=None(기본값) -> 다중검정 여부를 모르므로 이전 동작(개별 t-stat만 사용)과 동일
r_none = judge_signal(n_events=50, raw_mean_return=0.02, raw_t_stat=3.0,
                       alpha=0.01, alpha_t_stat=2.5, r_squared=0.4, t_threshold=T, bh_significant=None)
r_baseline = judge_signal(n_events=50, raw_mean_return=0.02, raw_t_stat=3.0,
                           alpha=0.01, alpha_t_stat=2.5, r_squared=0.4, t_threshold=T)
assert r_none["verdict"] == r_baseline["verdict"] == "채택"

print("judge_signal() 9개 분기 전부 통과 (기존 6개 + bh_significant 관련 3개, t_threshold=2.0 고정)")

# 10) t_threshold 기본값이 실제로 3.0으로 올라갔는지 확인 (Harvey/Liu/Zhu 2016 권고 반영)
#     |t|=2.5는 예전 기준(2.0)으로는 유의하지만 새 기본값(3.0)으로는 유의하지 않아야 함
r_default = judge_signal(n_events=50, raw_mean_return=0.02, raw_t_stat=2.5,
                          alpha=0.01, alpha_t_stat=3.5, r_squared=0.4)  # t_threshold 생략 -> 기본값 사용
assert r_default["verdict"] == "기각", r_default
assert "2.5" in r_default["reasons"][0] and "3.0" in r_default["reasons"][0], r_default["reasons"]

r_default_pass = judge_signal(n_events=50, raw_mean_return=0.02, raw_t_stat=3.5,
                               alpha=0.01, alpha_t_stat=3.2, r_squared=0.4)
assert r_default_pass["verdict"] == "채택", r_default_pass
print("[OK] t_threshold 기본값=3.0 확인 (|t|=2.5는 기각, |t|=3.5&3.2는 채택)")

# 7) judge_indicator() 통합 테스트 — event_study + event_alpha까지 실제로 엮어서 확인
np.random.seed(0)
n = 300
dates = pd.date_range("2023-01-01", periods=n, freq="D")
close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, n)))  # 약한 상승 드리프트
df = pd.DataFrame({"close": close}, index=dates)
# 10개마다 한 번씩 이벤트 발생 (표본 min_n_events=10을 넘기도록 충분히 많이)
df["evt"] = False
df.iloc[::10, df.columns.get_loc("evt")] = True

factor_dates = pd.date_range("2023-01-31", periods=12, freq="ME")
factor_df = pd.DataFrame({
    "SMB": np.random.normal(0, 0.01, 12),
    "HML": np.random.normal(0, 0.01, 12),
    "WML": np.random.normal(0, 0.01, 12),
}, index=factor_dates)

result = judge_indicator(df, "evt", direction=1, factor_df=factor_df, horizon=5)
print(result)
assert result["verdict"] in {"채택", "보류", "기각"}
assert result["n_events"] > 0

print("judge_indicator() 통합 실행 OK")
print("\nSMOKE TEST (decision) OK")
