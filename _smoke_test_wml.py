"""
factors/kr_fama_french.py의 WML(Carhart 1997) 그룹핑·집계 로직을 네트워크 없이
검증하는 스모크 테스트. fetch_fn(시가총액·종가)과 fetch_mom_fn(모멘텀 수익률)을
둘 다 가짜 함수로 주입해서 계산식만 확인한다.

시나리오: 6개 버킷(S/B × L/M/W) × 2종목 = 12종목.
- 그룹 배정용 모멘텀 수익률(mom_return, t0 시점에 "이미 알고 있는" 과거 형성기간
  수익률)은 L=-0.10, M=0.00, W=0.20으로 설계 — 이걸로 assign_momentum_groups가
  정확히 L/M/W를 되찾아내는지 확인한다(_smoke_test_factors.py와 같은 방식).
- 실제 t0->t1 구간의 forward return(WML이 최종적으로 측정하는 값)은 사이즈와
  무관하게 승자군 +8%p, 패자·중간군은 0%/+3%p로 설계했다.
  (S,L)=0.00 (S,M)=0.03 (S,W)=0.08
  (B,L)=0.00 (B,M)=0.03 (B,W)=0.08
  => WML = ((0.08+0.08)/2) - ((0.00+0.00)/2) = 0.08 이 나와야 한다.
"""
import pandas as pd
from factors.kr_fama_french import compute_wml

BUCKETS = {
    ("S", "L"): ["S_L_1", "S_L_2"],
    ("S", "M"): ["S_M_1", "S_M_2"],
    ("S", "W"): ["S_W_1", "S_W_2"],
    ("B", "L"): ["B_L_1", "B_L_2"],
    ("B", "M"): ["B_M_1", "B_M_2"],
    ("B", "W"): ["B_W_1", "B_W_2"],
}
SIZE_MKTCAP = {"S": 100.0, "B": 1000.0}
MOM_RETURN_FOR_GROUPING = {"L": -0.10, "M": 0.00, "W": 0.20}
FORWARD_RETURN_MAP = {
    ("S", "L"): 0.00, ("S", "M"): 0.03, ("S", "W"): 0.08,
    ("B", "L"): 0.00, ("B", "M"): 0.03, ("B", "W"): 0.08,
}
T0, T1 = "20260101", "20260201"


def fake_fetch(date: str) -> pd.DataFrame:
    """시가총액·종가·be_me(더미) 횡단면. be_me는 WML 계산엔 안 쓰지만
    assign_groups()가 컬럼 존재를 요구하므로 더미값(1.0)을 채워둔다."""
    rows = []
    for (size, mom), tickers in BUCKETS.items():
        for tk in tickers:
            close = 100.0 if date == T0 else 100.0 * (1 + FORWARD_RETURN_MAP[(size, mom)])
            rows.append(dict(ticker=tk, market="KOSPI", close=close,
                              mktcap=SIZE_MKTCAP[size], be_me=1.0))
    return pd.DataFrame(rows).set_index("ticker")


def fake_fetch_mom(date: str) -> pd.Series:
    """t0 시점에 이미 알려진 형성기간 모멘텀 수익률(그룹 배정용)."""
    rows = {}
    for (size, mom), tickers in BUCKETS.items():
        for tk in tickers:
            rows[tk] = MOM_RETURN_FOR_GROUPING[mom]
    return pd.Series(rows, name="mom_return")


result = compute_wml([T0, T1], fetch_fn=fake_fetch, fetch_mom_fn=fake_fetch_mom)
print(result)

assert not result.empty, "결과가 비어있으면 안 됨"
assert result.loc[T1, "n_stocks"] == 12
wml = result.loc[T1, "WML"]
assert abs(wml - 0.08) < 1e-9, f"WML 계산 오류: 기대값 0.08, 실제 {wml}"

print(f"\nWML={wml:.4f}(기대 0.08)")
print("SMOKE TEST (WML) OK")
