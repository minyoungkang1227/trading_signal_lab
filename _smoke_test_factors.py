"""
factors/kr_fama_french.py의 그룹핑·집계 로직을 네트워크 없이 검증하는 스모크 테스트.
pykrx 실호출(fetch_cross_section) 대신 가짜 fetch_fn을 주입해서, "SMB/HML 계산식이
의도한 대로 사이즈 효과와 가치 효과를 분리해내는가"만 순수 계산으로 확인한다.

시나리오: 6개 버킷(S/B × L/M/H)에 각 2종목씩 총 12종목을 만들고,
버킷별 수익률을 아래처럼 설계한다.
  (S,L)=+10%  (S,M)=+10%  (S,H)=+15%
  (B,L)=  0%  (B,M)=  0%  (B,H)=  5%
=> 순수 사이즈효과 +10%p, 순수 가치효과 +5%p가 S/B, L/M/H 양쪽에 균일하게 섞여있으므로
   SMB = (0.10+0.10+0.15)/3 - (0+0+0.05)/3 = 0.10  (가치효과는 상쇄되고 사이즈효과만 남아야 함)
   HML = (0.15+0.05)/2 - (0.10+0)/2           = 0.05  (사이즈효과는 상쇄되고 가치효과만 남아야 함)
"""
import pandas as pd
from factors.kr_fama_french import compute_smb_hml

BUCKETS = {
    ("S", "L"): ["S_L_1", "S_L_2"],
    ("S", "M"): ["S_M_1", "S_M_2"],
    ("S", "H"): ["S_H_1", "S_H_2"],
    ("B", "L"): ["B_L_1", "B_L_2"],
    ("B", "M"): ["B_M_1", "B_M_2"],
    ("B", "H"): ["B_H_1", "B_H_2"],
}
SIZE_MKTCAP = {"S": 100.0, "B": 1000.0}
BM_BEME = {"L": 0.3, "M": 0.8, "H": 1.5}
RETURN_MAP = {
    ("S", "L"): 0.10, ("S", "M"): 0.10, ("S", "H"): 0.15,
    ("B", "L"): 0.00, ("B", "M"): 0.00, ("B", "H"): 0.05,
}
T0, T1 = "20260101", "20260201"


def fake_fetch(date: str) -> pd.DataFrame:
    rows = []
    for (size, bm), tickers in BUCKETS.items():
        for tk in tickers:
            close = 100.0 if date == T0 else 100.0 * (1 + RETURN_MAP[(size, bm)])
            rows.append(dict(ticker=tk, market="KOSPI", close=close,
                              mktcap=SIZE_MKTCAP[size], be_me=BM_BEME[bm]))
    return pd.DataFrame(rows).set_index("ticker")


result = compute_smb_hml([T0, T1], fetch_fn=fake_fetch)
print(result)

assert not result.empty, "결과가 비어있으면 안 됨"
smb = result.loc[T1, "SMB"]
hml = result.loc[T1, "HML"]
assert result.loc[T1, "n_stocks"] == 12

assert abs(smb - 0.10) < 1e-9, f"SMB 계산 오류: 기대값 0.10, 실제 {smb}"
assert abs(hml - 0.05) < 1e-9, f"HML 계산 오류: 기대값 0.05, 실제 {hml}"

print(f"\nSMB={smb:.4f}(기대 0.10), HML={hml:.4f}(기대 0.05)")
print("SMOKE TEST (factors) OK")
