"""
run_backtest.py의 argparse 기본값과 1:1로 맞춘 DEFAULTS/build_args.

app.py의 같은 이름 함수/딕셔너리와 패턴이 완전히 동일하다 — 의도적인 독립 복제다
(app.py는 streamlit 런타임 밖에서 import하면 st.set_page_config() 등이 즉시
실행돼 에러가 나므로, GitHub Actions처럼 streamlit 없이 도는 스크립트에서는
app.py를 그대로 import할 수 없다). run_backtest.py의 CLI에 새 --옵션이 추가되면
app.py의 DEFAULTS와 이 파일의 DEFAULTS 둘 다 같이 고쳐야 한다 — 이 파일 docstring과
app.py docstring 양쪽에 동일하게 남겨둔 유지보수 포인트.

watchlist.json 형식(리스트, 각 항목 하나가 감시할 종목 하나):
  krx  : {"market": "krx", "symbol": "005930", "label": "삼성전자",
          "start": "20230101" (생략 가능, 기본 2년 전), "end": 생략(기본 오늘)}
  upbit: {"market": "upbit", "symbol": "KRW-BTC", "label": "비트코인",
          "unit": "day" (생략 가능), "count": 1000 (생략 가능)}
"""
from __future__ import annotations

import json
import types
from datetime import datetime, timedelta
from pathlib import Path

DEFAULTS = dict(
    unit="day", minute_unit=60, count=1000, start=None, end=None,
    horizons="1,3,5,10,20",
    id_vol_mode="multiple", id_vol_mult=2.0, id_z_threshold=1.96,
    mom_filter=False, mom_lookback=126, mom_window=252, mom_threshold=0.7,
    entry_lag=0, entry_col=None, cost_bps=0.0, slippage_bps=0.0,
    newey_west=False, drop_overlapping=False,
    liq_filter=False, liq_window=20, liq_percentile_window=252, liq_threshold=0.9,
    dynamic_slippage=False, max_extra_slippage_bps=50.0,
    fdr=0.10, factor_csv=None, factor_cols=None, t_threshold=3.0,
    big_sales_len=7, vp_bins=20, vp_freq="W",
    combo_body_min_pct=55.0, combo_wick_max_pct=25.0, combo_vol_len=20,
    combo_vol_mult=1.6, combo_ma_len=50, combo_ma_type="EMA", combo_score_threshold=0.62,
    log_executions=False, execution_log_path=None, output="output",
)

# krx 종목인데 watchlist.json에 start/end를 안 적었을 때 기본으로 쓸 조회 기간.
# 콤보필터 ma_len=50, vol_len=20 등 롤링 계산에 여유 있게 쌓이도록 2년으로 잡는다.
_DEFAULT_LOOKBACK_DAYS = 730


def _today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def build_args(**overrides) -> types.SimpleNamespace:
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


def args_for_watch_item(item: dict) -> types.SimpleNamespace:
    """watchlist.json의 항목 하나(dict)를 run_backtest.load_data/run에 넘길
    args로 변환한다. market별로 필요한 필드만 채우고 나머지는 DEFAULTS를 쓴다."""
    market = item["market"]
    symbol = item["symbol"]

    if market == "krx":
        end = item.get("end") or _today_str()
        if item.get("start"):
            start = item["start"]
        else:
            start_dt = datetime.strptime(end, "%Y%m%d") - timedelta(days=_DEFAULT_LOOKBACK_DAYS)
            start = start_dt.strftime("%Y%m%d")
        return build_args(market="krx", symbol=symbol, start=start, end=end)

    if market == "upbit":
        return build_args(
            market="upbit", symbol=symbol,
            unit=item.get("unit", DEFAULTS["unit"]),
            minute_unit=item.get("minute_unit", DEFAULTS["minute_unit"]),
            count=item.get("count", DEFAULTS["count"]),
        )

    raise ValueError(f"알 수 없는 market: {market!r} (watchlist 항목: {item})")


def load_watchlist(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"감시 리스트 파일이 없습니다: {path} — alerts/watchlist.json을 먼저 만드세요."
        )
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        if "market" not in item or "symbol" not in item:
            raise ValueError(f"watchlist 항목에 market/symbol이 빠졌습니다: {item}")
    return items
