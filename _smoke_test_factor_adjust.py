"""
backtest/engine.py::factor_adjust()의 OLS 회귀 로직을 네트워크 없이 검증한다.
합성 팩터 시계열과, 그 팩터들의 정확한 선형결합으로 만든 이벤트 수익률을 넣으면
회귀가 alpha·베타를 원래 넣은 값 그대로 복원해야 하고 R^2는 1에 가까워야 한다
(노이즈가 없는 완전선형 관계이므로).
"""
import numpy as np
import pandas as pd
from backtest.engine import factor_adjust, event_alpha

N = 40
DATES = pd.date_range("2020-01-31", periods=N, freq="ME")

SMB = 0.01 * (np.arange(N) % 5 - 2)
HML = 0.01 * (np.arange(N) % 3 - 1)
WML = 0.01 * (np.arange(N) % 4 - 1.5)

TRUE_ALPHA, TRUE_B_SMB, TRUE_B_HML, TRUE_B_WML = 0.015, 0.6, -0.4, 0.25
EVENT_RETURNS = TRUE_ALPHA + TRUE_B_SMB * SMB + TRUE_B_HML * HML + TRUE_B_WML * WML

factor_df = pd.DataFrame({"SMB": SMB, "HML": HML, "WML": WML}, index=DATES)

# 1) 저수준 factor_adjust() 직접 호출
result = factor_adjust(DATES, EVENT_RETURNS, factor_df)
print(result)

assert result["n_obs"] == N
assert abs(result["alpha"] - TRUE_ALPHA) < 1e-9, f"alpha 복원 실패: {result['alpha']}"
assert abs(result["betas"]["SMB"][0] - TRUE_B_SMB) < 1e-9
assert abs(result["betas"]["HML"][0] - TRUE_B_HML) < 1e-9
assert abs(result["betas"]["WML"][0] - TRUE_B_WML) < 1e-9
assert result["r_squared"] > 0.999, f"R^2 낮음: {result['r_squared']}"

# 2) 고수준 event_alpha() — df+event_col 인터페이스로도 같은 결과가 나오는지 확인
# forward_return(close,1)[t] = close[t+1]/close[t]-1 이 EVENT_RETURNS[t]와 정확히
# 일치하도록 close를 역산한다: close[0]=100, close[t+1]=close[t]*(1+EVENT_RETURNS[t]).
closes = 100.0 * np.concatenate([[1.0], np.cumprod(1 + EVENT_RETURNS[:-1])])
df = pd.DataFrame({"close": closes}, index=DATES)
df["dummy_event"] = True
# horizon=1의 순방향수익률이 EVENT_RETURNS와 정확히 일치하도록 close를 역산했으므로
# 마지막 이벤트는 다음 봉이 없어 자동으로 빠진다(n_obs가 1개 준다).
result2 = event_alpha(df, "dummy_event", factor_df, horizon=1)
print(result2)
assert result2["n_obs"] == N - 1
assert abs(result2["alpha"] - TRUE_ALPHA) < 1e-6, f"event_alpha alpha 불일치: {result2['alpha']}"

# 3) 표본 부족 시 NaN 반환하는지 확인 (파라미터 4개인데 관측치 2개)
tiny = factor_adjust(DATES[:2], EVENT_RETURNS[:2], factor_df)
assert tiny["n_obs"] == 2
assert np.isnan(tiny["alpha"])

print("\nSMOKE TEST (factor_adjust) OK")
