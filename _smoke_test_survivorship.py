"""
factors/kr_fama_french.py의 _value_weighted_return() 생존 편향(survivorship bias)
처리를 검증하는 스모크 테스트.

문제: t0(리밸런싱 시점)에는 존재하던 종목이 t1(다음 리밸런싱 시점)에는 상장폐지·
거래정지 등으로 종가 조회가 안 될 수 있다. 이전 동작(delisted_return=None)은 그런
종목을 조용히 빼고 남은 종목끼리만 가중치를 재정규화하는데, 상장폐지는 거의 항상
큰 손실로 끝나므로 이렇게 빼면 포트폴리오 수익률이 실제보다 좋게(위로) 편향된다.
delisted_return을 지정하면(예: -1.0=전손) 그 손실을 강제로 반영해 편향을 없앤다.

시나리오 1 (저수준 함수 직접 검증): 두 종목 A(시총 100, 생존, +20%), B(시총 100,
t1에 종가 없음=상장폐지)로 구성된 포트폴리오.
  - delisted_return=None: B를 빼고 A만 남음 -> 수익률 = +20%
  - delisted_return=-1.0: A(+20%)와 B(-100%)를 시총가중(50:50) 평균
                          -> 수익률 = 0.5*0.20 + 0.5*(-1.0) = -0.40
"""
import pandas as pd
from factors.kr_fama_french import _value_weighted_return, compute_smb_hml

# --- 시나리오 1: 저수준 함수 ---
weight = pd.Series({"A": 100.0, "B": 100.0})
close_t0 = pd.Series({"A": 100.0, "B": 100.0})
close_t1 = pd.Series({"A": 120.0})  # B는 t1에 종가 없음 (상장폐지로 가정)

ret_drop = _value_weighted_return(weight, close_t0, close_t1)
assert abs(ret_drop - 0.20) < 1e-9, f"기본(None) 모드는 B를 빼고 A만 남아야 함: {ret_drop}"
print(f"[OK] delisted_return=None (기존 동작): {ret_drop:.4f} (기대 0.20, B 조용히 제외)")

ret_total_loss = _value_weighted_return(weight, close_t0, close_t1, delisted_return=-1.0)
expected = 0.5 * 0.20 + 0.5 * (-1.0)
assert abs(ret_total_loss - expected) < 1e-9, f"전손 반영 모드 계산 오류: {ret_total_loss} vs {expected}"
print(f"[OK] delisted_return=-1.0 (전손 반영): {ret_total_loss:.4f} (기대 {expected:.4f})")

assert ret_total_loss < ret_drop, "상장폐지 손실을 반영하면 반드시 수익률이 낮아져야 함(생존 편향 방향 확인)"
print("[OK] 전손 반영 시 수익률이 기존(제외) 대비 낮게 나옴 -> 생존 편향 방향 일치")

# --- 시나리오 2: compute_smb_hml() 통합 — (S,H) 버킷의 종목 하나가 t1에 사라짐 ---
BUCKETS = {
    ("S", "L"): ["S_L_1", "S_L_2"],
    ("S", "M"): ["S_M_1", "S_M_2"],
    ("S", "H"): ["S_H_1", "S_H_2"],  # S_H_2가 t1에 상장폐지
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
DELISTED = "S_H_2"
T0, T1 = "20260101", "20260201"


def fake_fetch(date: str) -> pd.DataFrame:
    rows = []
    for (size, bm), tickers in BUCKETS.items():
        for tk in tickers:
            if date == T1 and tk == DELISTED:
                continue  # 상장폐지: t1 스냅샷에서 아예 빠짐
            close = 100.0 if date == T0 else 100.0 * (1 + RETURN_MAP[(size, bm)])
            rows.append(dict(ticker=tk, market="KOSPI", close=close,
                              mktcap=SIZE_MKTCAP[size], be_me=BM_BEME[bm]))
    return pd.DataFrame(rows).set_index("ticker")


result_drop = compute_smb_hml([T0, T1], fetch_fn=fake_fetch)
result_bias_fixed = compute_smb_hml([T0, T1], fetch_fn=fake_fetch, delisted_return=-1.0)

smb_drop, hml_drop = result_drop.loc[T1, "SMB"], result_drop.loc[T1, "HML"]
smb_fixed, hml_fixed = result_bias_fixed.loc[T1, "SMB"], result_bias_fixed.loc[T1, "HML"]

print(f"\n[제외 모드]   SMB={smb_drop:.4f} HML={hml_drop:.4f}")
print(f"[전손 반영 모드] SMB={smb_fixed:.4f} HML={hml_fixed:.4f}")

# (S,H) 버킷 수익률이 낮아지므로 -> HML(H 그룹에 의존)도, SMB(S 그룹에 의존)도 낮아져야 함
assert smb_fixed < smb_drop, "생존 편향 보정 후 SMB가 낮아지지 않음"
assert hml_fixed < hml_drop, "생존 편향 보정 후 HML이 낮아지지 않음"
print("[OK] 상장폐지 손실을 반영하면 SMB·HML 모두 더 낮게(더 보수적으로) 나옴")

print("\nSMOKE TEST (survivorship bias) OK")
