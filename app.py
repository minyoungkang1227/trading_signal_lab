"""
trading_signal_lab을 매번 CLI로 돌리지 않고 브라우저에서 바로 돌려보는 Streamlit 대시보드.

[설계 원칙] run_backtest.py의 실제 로직(load_data/run/run_judgment)을 그대로 import해서
재사용한다 — 이 파일 안에서 백테스트 로직을 다시 구현하지 않는다. CLI와 대시보드가
서로 다른 결과를 낼 위험(로직이 두 곳에 흩어져 하나만 고치는 실수)을 원천 차단하기 위함.

run_backtest.main()의 argparse 결과를 흉내낸 SimpleNamespace를 직접 만들어서
run_backtest.load_data()/run()/run_judgment()에 그대로 넘긴다 — CLI 플래그 이름과
1:1 대응이라 CLI에 새 옵션이 추가되면 build_args()의 DEFAULTS에도 같이 추가해야
깨지지 않는다(이 파일의 유일한 유지보수 포인트).

사용법: streamlit run app.py
"""
from __future__ import annotations

import types

import pandas as pd
import streamlit as st

import run_backtest as rb
from backtest.engine import SIGNAL_SPECS

# run_backtest.py main()의 argparse 기본값과 1:1로 맞춘 기본 설정.
# CLI에 새 --옵션이 추가되면 여기에도 같은 dest/default를 추가해야 함.
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


def build_args(**overrides) -> types.SimpleNamespace:
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


st.set_page_config(page_title="trading_signal_lab 대시보드", layout="wide")
st.title("trading_signal_lab — 신호 검증 대시보드")
st.caption(
    "run_backtest.py를 매번 터미널에서 돌리는 대신 여기서 종목/파라미터만 바꿔가며 "
    "이벤트 스터디 결과를 바로 확인한다. 로직은 run_backtest.py와 완전히 동일 — "
    "이 페이지는 그 함수들을 그대로 호출하는 얇은 껍데기다."
)

with st.sidebar:
    st.header("데이터")
    market = st.selectbox("시장", ["krx", "upbit"], index=0)
    symbol = st.text_input(
        "종목코드", value="005930" if market == "krx" else "KRW-BTC",
        help="krx: '005930' 형식 / upbit: 'KRW-BTC' 형식",
    )
    if market == "krx":
        start = st.text_input("시작일 (YYYYMMDD)", value="20200101")
        end = st.text_input("종료일 (YYYYMMDD)", value="20260901")
        count, unit, minute_unit = DEFAULTS["count"], DEFAULTS["unit"], DEFAULTS["minute_unit"]
    else:
        unit = st.selectbox("봉 단위", ["day", "week", "minute"], index=0)
        minute_unit = st.number_input("분봉 단위(unit=minute일 때)", value=60, step=1) if unit == "minute" else 60
        count = st.number_input("캔들 개수", value=1000, step=100)
        start = end = None

    horizons = st.text_input("검증할 보유기간(봉수, 콤마구분)", value="1,3,5,10,20")

    st.header("메타라벨링 1단계")
    log_executions = st.checkbox(
        "체결 로그 기록 (--log-executions)", value=False,
        help="이번 실행의 신호 이벤트를 signal_id 단위로 output/execution_log.csv에 누적 기록. "
             "exit_reason은 전부 'horizon_exit'(실제 리스크관리 청산 사유 아님).",
    )

    with st.expander("고급 설정 (체결 가정 / 다중검정 보정)"):
        entry_lag = st.number_input("entry_lag (진입 지연 봉수)", value=0, step=1)
        entry_col = st.selectbox("entry_col", [None, "open", "close"], index=0)
        cost_bps = st.number_input("cost_bps (왕복 수수료, bp)", value=0.0, step=1.0)
        slippage_bps = st.number_input("slippage_bps (왕복 슬리피지, bp)", value=0.0, step=1.0)
        newey_west = st.checkbox("Newey-West HAC 보정", value=False)
        drop_overlapping = st.checkbox("겹치는 이벤트 제거", value=False)
        fdr = st.number_input("FDR (Benjamini-Hochberg)", value=0.10, min_value=0.01, max_value=0.5, step=0.01)
        t_threshold = st.number_input("t_threshold (판정 기준)", value=3.0, step=0.1)

    with st.expander("콤보필터 파라미터 (candle_volume_entry_filter.pine 포팅)"):
        combo_body_min_pct = st.number_input("body_min_pct", value=55.0)
        combo_wick_max_pct = st.number_input("wick_max_pct", value=25.0)
        combo_vol_len = st.number_input("vol_len", value=20, step=1)
        combo_vol_mult = st.number_input("vol_mult", value=1.6)
        combo_ma_len = st.number_input("ma_len", value=50, step=1)
        combo_ma_type = st.selectbox("ma_type", ["EMA", "SMA", "WMA"], index=0)
        combo_score_threshold = st.number_input(
            "score_threshold", value=0.62, min_value=0.0, max_value=1.0,
            help="원본 Pine 채택값(0.62) — event_study로 재검증 전까지는 참고용으로 취급할 것",
        )

    run_clicked = st.button("실행", type="primary", use_container_width=True)

if not run_clicked:
    st.info("왼쪽에서 종목/파라미터를 설정하고 [실행]을 눌러줘.")
    st.stop()

args = build_args(
    market=market, symbol=symbol, unit=unit, minute_unit=minute_unit, count=count,
    start=start, end=end, horizons=horizons,
    entry_lag=entry_lag, entry_col=entry_col, cost_bps=cost_bps, slippage_bps=slippage_bps,
    newey_west=newey_west, drop_overlapping=drop_overlapping, fdr=fdr, t_threshold=t_threshold,
    combo_body_min_pct=combo_body_min_pct, combo_wick_max_pct=combo_wick_max_pct,
    combo_vol_len=combo_vol_len, combo_vol_mult=combo_vol_mult, combo_ma_len=combo_ma_len,
    combo_ma_type=combo_ma_type, combo_score_threshold=combo_score_threshold,
    log_executions=log_executions,
)

with st.spinner(f"{market}/{symbol} 데이터 수집 중..."):
    try:
        df = rb.load_data(args)
    except SystemExit as e:
        st.error(f"데이터 수집 실패: {e}")
        st.stop()
    except Exception as e:
        st.error(f"데이터 수집 중 오류: {e}")
        st.stop()

st.success(f"{len(df)}봉 수집 완료 ({df.index.min().date()} ~ {df.index.max().date()})")

with st.spinner("이벤트 스터디 계산 중..."):
    summary, results, dfs_by_indicator, slippage_arg = rb.run(df, args)

if summary.empty:
    st.warning("검증된 이벤트가 없습니다 (신호가 한 번도 발생하지 않았을 수 있습니다).")
    st.stop()

horizon_list = [int(h) for h in horizons.split(",")]
main_horizon = st.selectbox(
    "랭킹에 쓸 보유기간", horizon_list, index=len(horizon_list) // 2,
)

sub = summary[summary["horizon"] == main_horizon].sort_values("t_stat", ascending=False)
st.subheader(f"보유기간 {main_horizon}봉 기준 t-stat 랭킹")
st.caption(
    "|t-stat|>=2면 눈여겨볼 만한 신호, n_events<10이면 표본 부족. significant_bh는 "
    "이번 실행의 전체 신호x방향x보유기간 조합을 한꺼번에 BH 절차로 보정한 뒤에도 "
    "유의한 것만 True — 개별 t_stat만 보고 판단하면 우연히 유의하게 나온 신호를 "
    "걸러내지 못할 수 있다."
)
st.dataframe(
    sub[["signal", "direction", "n_events", "mean_return_pct", "win_rate_pct",
         "t_stat", "p_value", "significant_bh"]].reset_index(drop=True),
    use_container_width=True,
)

st.subheader("신호 발생 후 평균 누적수익률 경로")
curve_df = pd.DataFrame({
    label: res.curve for label, res in results.items() if res.n_events > 0
})
if not curve_df.empty:
    st.line_chart(curve_df * 100)
else:
    st.caption("플롯할 이벤트가 없습니다.")

if log_executions:
    from backtest.execution_log import log_all_signals
    from pathlib import Path

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.execution_log_path or str(out_dir / "execution_log.csv")
    n_new = log_all_signals(
        dfs_by_indicator, symbol=args.symbol, horizon=main_horizon, path=log_path,
        entry_lag=args.entry_lag, entry_col=args.entry_col, cost_bps=args.cost_bps,
        slippage_bps=slippage_arg, drop_overlapping=args.drop_overlapping,
    )
    st.success(
        f"체결 로그 {log_path}에 {n_new}건 신규 기록 (보유기간={main_horizon}봉 기준, "
        f"exit_reason은 전부 'horizon_exit' — 실제 리스크관리 청산 사유 아님)"
    )
    if Path(log_path).exists():
        log_df = pd.read_csv(log_path)
        with st.expander(f"누적 체결 로그 미리보기 ({len(log_df)}행)"):
            st.dataframe(log_df.tail(50), use_container_width=True)
        st.download_button(
            "execution_log.csv 다운로드", data=Path(log_path).read_bytes(),
            file_name="execution_log.csv", mime="text/csv",
        )

st.caption(
    "주의: 이 화면은 과거 통계 기반 신호 요약이지 투자자문이 아닙니다. "
    "n_events가 작거나(특히 10 미만) significant_bh가 False면 결론으로 쓰기엔 이릅니다."
)
