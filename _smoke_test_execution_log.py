"""
execution_log.py 스모크 테스트.
(1) 손계산: 단일 이벤트에 대해 entry/exit price, pnl, exit_reason을 손으로 계산한
    값과 비교.
(2) signal_id 결정성: 같은 (symbol, date, label) -> 같은 id, 다른 입력 -> 다른 id.
(3) log_all_signals + append_to_log: SIGNAL_SPECS 전체 순회 -> CSV 저장 ->
    같은 데이터로 재실행 시 중복 없이 0행만 추가되는지(재실행 안전성) 확인.
"""
import os
import tempfile

import numpy as np
import pandas as pd

from backtest.execution_log import (
    generate_signal_id, log_event_study_trades, append_to_log, log_all_signals,
)

# ---- (1) 손계산 단일 이벤트 ----
dates = pd.date_range("2024-01-01", periods=10, freq="D")
close = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
df = pd.DataFrame({"close": close}, index=dates)
df.index.name = "date"
df["sig"] = False
df.loc[dates[2], "sig"] = True  # 2024-01-03 하루만 이벤트

rows = log_event_study_trades(df, "sig", direction=1, label="테스트신호",
                               symbol="TEST", horizon=3)
assert len(rows) == 1, f"기대 이벤트 1건, 실제 {len(rows)}건"
r = rows.iloc[0]

expected_entry = 102.0  # close[idx=2]
expected_exit = 105.0   # close[idx=2+3=5]
expected_pnl = expected_exit / expected_entry - 1  # 0.029411...

assert abs(r["entry_price"] - expected_entry) < 1e-9, r["entry_price"]
assert abs(r["exit_price"] - expected_exit) < 1e-9, r["exit_price"]
assert abs(r["pnl"] - expected_pnl) < 1e-9, (r["pnl"], expected_pnl)
assert r["exit_reason"] == "horizon_exit"
assert r["basis"] == "backtest_event_study"
assert r["symbol"] == "TEST" and r["label"] == "테스트신호" and r["direction"] == 1
assert r["date"] == "2024-01-03"
print("손계산 단일 이벤트 검증 통과:", dict(r))

# ---- (2) signal_id 결정성 ----
id_a = generate_signal_id("TEST", "2024-01-03", "테스트신호")
id_b = generate_signal_id("TEST", "2024-01-03", "테스트신호")
id_c = generate_signal_id("TEST", "2024-01-04", "테스트신호")  # 날짜만 다름
id_d = generate_signal_id("TEST2", "2024-01-03", "테스트신호")  # 심볼만 다름
assert id_a == id_b, "같은 입력인데 id가 다름"
assert id_a != id_c and id_a != id_d, "다른 입력인데 id가 같음(해시 충돌 의심)"
assert id_a == r["signal_id"], "log_event_study_trades가 만든 id와 직접 호출 결과가 다름"
print("signal_id 결정성 검증 통과:", id_a)

# ---- (3) log_all_signals + append_to_log 재실행 안전성 ----
np.random.seed(42)
n = 300
dates2 = pd.date_range("2023-01-01", periods=n, freq="D")
ret = np.random.normal(0, 0.015, n)
close2 = 100 * np.exp(np.cumsum(ret))
big_df = pd.DataFrame({"close": close2}, index=dates2)
big_df.index.name = "date"
big_df["bull_shift"] = False
# 10번째 봉마다 이벤트가 뜬다고 가정(합성 데이터, 손으로 센 이벤트 수와 비교 가능)
event_positions = list(range(5, n - 25, 10))
big_df.loc[big_df.index[event_positions], "bull_shift"] = True
expected_n_events = len(event_positions)

dfs_by_indicator = {"institutional_displacement": big_df}

with tempfile.TemporaryDirectory() as tmp:
    log_path = os.path.join(tmp, "execution_log.csv")

    n_new_1 = log_all_signals(dfs_by_indicator, symbol="005930", horizon=5, path=log_path)
    assert n_new_1 == expected_n_events, f"1차 기록: 기대 {expected_n_events}건, 실제 {n_new_1}건"
    assert os.path.exists(log_path)

    saved = pd.read_csv(log_path)
    assert len(saved) == expected_n_events
    assert set(saved["signal_id"]) == set(saved["signal_id"].unique())  # 자체 중복 없음
    assert (saved["label"] == "기관캔들-강세").all()

    # 같은 데이터로 재실행 -> 전부 중복이라 0건만 추가돼야 함
    n_new_2 = log_all_signals(dfs_by_indicator, symbol="005930", horizon=5, path=log_path)
    assert n_new_2 == 0, f"재실행 시 0건이어야 하는데 {n_new_2}건 추가됨(중복 방지 실패)"

    saved_again = pd.read_csv(log_path)
    assert len(saved_again) == expected_n_events, "재실행 후 행 수가 바뀌면 안 됨"

    print(f"log_all_signals + append_to_log 재실행 안전성 검증 통과: "
          f"1차 {n_new_1}건 기록, 재실행 {n_new_2}건 추가(중복 방지 확인)")

print("\nSMOKE TEST OK (execution_log)")
