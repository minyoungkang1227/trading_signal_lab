"""
trading_signal_lab을 매번 CLI로 돌리지 않고 브라우저에서 바로 돌려보는 Streamlit 대시보드.

[설계 원칙] run_backtest.py의 실제 로직(load_data/run/run_judgment)을 그대로 import해서
재사용한다 — 이 파일 안에서 백테스트 로직을 다시 구현하지 않는다. CLI와 대시보드가
서로 다른 결과를 낼 위험(로직이 두 곳에 흩어져 하나만 고치는 실수)을 원천 차단하기 위함.

run_backtest.main()의 argparse 결과를 흉내낸 SimpleNamespace를 직접 만들어서
run_backtest.load_data()/run()/run_judgment()에 그대로 넘긴다 — CLI 플래그 이름과
1:1 대응이라 CLI에 새 옵션이 추가되면 build_args()의 DEFAULTS에도 같이 추가해야
깨지지 않는다(이 파일의 유일한 유지보수 포인트).

이 파일은 화면 문구(한글 라벨, 설명 문구)만 다루고 계산 로직에는 전혀 손대지 않는다
— 표시용 매핑(DIRECTION_LABELS, COLUMN_LABELS 등)은 전부 화면에 보여주기 직전에만
적용되고, run_backtest.py로 넘어가는 값(args)이나 CLI 동작에는 영향을 주지 않는다.

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

# 결과 화면에서만 쓰는 한글 라벨 매핑 (run_backtest.py/backtest/engine.py의 실제
# 컬럼명·값은 그대로 두고, 화면에 보여줄 때만 바꿔치기한다).
COLUMN_LABELS = {
    "signal": "신호",
    "direction": "방향",
    "n_events": "표본수",
    "mean_return_pct": "평균수익률(%)",
    "win_rate_pct": "승률(%)",
    "t_stat": "t값",
    "p_value": "p값",
    "significant_bh": "통계적 유의 (BH보정)",
}
DIRECTION_LABELS = {"LONG": "상승(롱)", "SHORT": "하락(숏)"}


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

with st.expander("이 페이지가 하는 일이 궁금하다면 (처음이면 한 번 읽어보세요)"):
    st.markdown(
        "- **무엇을 하나요**: 특정 조건(예: 거래량이 평소보다 많이 터졌다, 두 이동평균선이 "
        "교차했다 등)이 발생했을 때마다 그 시점을 '신호'로 표시해두고, 그 이후 일정 기간이 "
        "지났을 때 실제로 주가가 어떻게 움직였는지를 과거 데이터로 전부 모아서 통계적으로 "
        "요약해줍니다. 이걸 흔히 '이벤트 스터디'라고 부릅니다.\n"
        "- **왜 필요한가요**: \"이 신호, 왠지 잘 맞는 것 같다\"는 감이 아니라, 과거에 그 신호가 "
        "몇 번 나왔고(표본수) 그때마다 평균 몇 % 움직였고(평균수익률) 그게 우연이 아니라고 "
        "볼 만큼 통계적으로 믿을 만한지(t값·p값)를 숫자로 확인하기 위함입니다.\n"
        "- **주의할 점**: 과거에 잘 맞았다고 미래에도 그대로 맞는다는 보장은 없습니다. 표본수가 "
        "너무 적거나(10개 미만) 통계적 유의성을 통과하지 못한 신호는 아직 결론으로 쓰기엔 "
        "이릅니다. 이 화면은 투자 자문이 아니라 검증 도구입니다."
    )

with st.sidebar:
    st.header("1. 데이터 불러오기")
    st.caption("어떤 종목을, 어느 기간만큼 가져와서 검증할지 정합니다.")
    market = st.selectbox(
        "시장", ["krx", "upbit"], index=0,
        format_func=lambda v: {"krx": "국내 주식 (KRX)", "upbit": "업비트 코인"}[v],
    )
    symbol = st.text_input(
        "종목코드", value="005930" if market == "krx" else "KRW-BTC",
        help="국내 주식은 '005930'(삼성전자)처럼 6자리 종목코드, "
             "업비트는 'KRW-BTC'처럼 '원화마켓-코인심볼' 형식으로 입력하세요.",
    )
    if market == "krx":
        start = st.text_input(
            "시작일 (YYYYMMDD)", value="20200101",
            help="데이터를 가져올 시작 날짜. 예: 2020년 1월 1일 → 20200101",
        )
        end = st.text_input(
            "종료일 (YYYYMMDD)", value="20260901",
            help="데이터를 가져올 마지막 날짜.",
        )
        count, unit, minute_unit = DEFAULTS["count"], DEFAULTS["unit"], DEFAULTS["minute_unit"]
    else:
        unit = st.selectbox(
            "봉 단위", ["day", "week", "minute"], index=0,
            format_func=lambda v: {"day": "일봉", "week": "주봉", "minute": "분봉"}[v],
        )
        minute_unit = st.number_input(
            "분봉 단위(분)", value=60, step=1,
            help="봉 단위를 '분봉'으로 선택했을 때만 사용됩니다. 예: 60이면 60분봉.",
        ) if unit == "minute" else 60
        count = st.number_input(
            "캔들 개수", value=1000, step=100,
            help="가장 최근 캔들부터 몇 개를 가져올지.",
        )
        start = end = None

    horizons = st.text_input(
        "검증할 보유기간 (봉 수, 콤마로 구분)", value="1,3,5,10,20",
        help="신호가 뜬 뒤 1봉, 3봉, 5봉... 뒤의 수익률을 각각 확인합니다. "
             "일봉 기준이면 '5'는 신호 발생 5거래일 뒤라는 뜻입니다.",
    )

    st.header("2. 체결 기록 (선택)")
    st.caption(
        "메타라벨링(나중에 어떤 신호가 더 믿을 만한지 머신러닝으로 다시 거르는 작업)을 "
        "위한 준비 단계입니다. 지금 당장 필요 없다면 꺼둔 채로 실행해도 결과 확인에는 "
        "아무 지장이 없습니다."
    )
    log_executions = st.checkbox(
        "이번 실행의 신호를 체결 로그 파일에 기록하기", value=False,
        help="이번 실행에서 나온 신호 하나하나를 output/execution_log.csv 파일에 "
             "계속 누적해서 쌓아둡니다. 나중에 메타라벨링 모델을 학습시킬 때 쓸 원재료를 "
             "미리 모아두는 용도입니다. 참고: 청산 사유(exit_reason)는 전부 '보유기간 "
             "만료(horizon_exit)'로 기록되며, 실제 손절/익절 로직이 적용된 건 아닙니다.",
    )

    with st.expander("3. 고급 설정 — 체결 가정 · 다중검정 보정"):
        st.caption(
            "실제로 거래했다면 어떤 조건으로 진입했을지, 그리고 여러 신호를 한꺼번에 "
            "비교할 때 생기는 통계적 착시를 어떻게 보정할지에 대한 설정입니다. 잘 모르겠으면 "
            "기본값 그대로 두어도 됩니다."
        )
        entry_lag = st.number_input(
            "진입 지연 (봉 수)", value=0, step=1,
            help="신호가 뜬 당일 바로 진입하지 않고 몇 봉 뒤에 진입할지. 0이면 신호 당일 진입.",
        )
        entry_col = st.selectbox(
            "진입 가격 기준", [None, "open", "close"], index=0,
            format_func=lambda v: {
                None: "기본값 (진입 지연만 반영)", "open": "시가 기준", "close": "종가 기준",
            }[v],
            help="진입 가격을 그날의 시가로 할지 종가로 할지. 기본값은 진입 지연 설정만 반영합니다.",
        )
        cost_bps = st.number_input(
            "거래수수료 (왕복, bp)", value=0.0, step=1.0,
            help="사고팔 때 드는 수수료를 bp(1bp=0.01%) 단위로 가정해서 수익률에서 뺍니다.",
        )
        slippage_bps = st.number_input(
            "슬리피지 (왕복, bp)", value=0.0, step=1.0,
            help="주문가와 실제 체결가의 차이(미끄러짐)를 bp 단위로 가정해서 수익률에서 뺍니다.",
        )
        newey_west = st.checkbox(
            "Newey-West 표준오차 보정 적용", value=False,
            help="수익률 간에 자기상관이 있으면 t값이 실제보다 부풀려질 수 있는데, 이를 "
                 "보정해서 더 보수적인(엄격한) t값을 계산합니다.",
        )
        drop_overlapping = st.checkbox(
            "겹치는 이벤트는 하나만 남기기", value=False,
            help="같은 보유기간 안에 신호가 또 뜨면 서로 겹치는 구간이 생겨 표본이 "
                 "독립적이지 않게 되는데, 이를 막기 위해 겹치는 신호를 솎아냅니다.",
        )
        fdr = st.number_input(
            "다중검정 허용 오류율 (FDR)", value=0.10, min_value=0.01, max_value=0.5, step=0.01,
            help="여러 신호 x 방향 x 보유기간 조합을 한꺼번에 비교하면 그중 일부는 우연히 "
                 "유의하게 나올 수 있습니다(Benjamini-Hochberg 보정). 값이 작을수록 더 "
                 "엄격한 기준으로 '진짜 유의한' 신호만 통과시킵니다.",
        )
        t_threshold = st.number_input(
            "판정 기준 t값", value=3.0, step=0.1,
            help="신호를 '쓸 만하다'고 판정할 때 기준으로 삼을 t값의 최소 크기.",
        )

    with st.expander("4. 콤보필터 파라미터 — 캔들+거래량 진입 필터"):
        st.caption(
            "원본 Pine Script 지표(candle_volume_entry_filter)를 그대로 포팅한 콤보필터의 "
            "세부 조건입니다. '몸통이 크고 꼬리는 짧은 캔들 + 평소보다 많은 거래량 + 추세 "
            "방향 일치'를 모두 만족할 때만 신호로 인정합니다."
        )
        combo_body_min_pct = st.number_input(
            "캔들 몸통 비율 최소값 (%)", value=55.0,
            help="캔들 전체 길이(고가-저가) 대비 몸통(시가-종가)이 최소 몇 % 이상이어야 "
                 "'힘있는' 캔들로 인정할지.",
        )
        combo_wick_max_pct = st.number_input(
            "꼬리 비율 최대값 (%)", value=25.0,
            help="위/아래 꼬리가 캔들 전체 길이의 몇 %를 넘으면 신호에서 제외할지.",
        )
        combo_vol_len = st.number_input(
            "거래량 비교 구간 (봉 수)", value=20, step=1,
            help="최근 거래량이 평소보다 많은지 비교할 때, 몇 봉 평균과 비교할지.",
        )
        combo_vol_mult = st.number_input(
            "거래량 배수 기준", value=1.6,
            help="평균 거래량의 몇 배 이상 터져야 '거래량 확인됨'으로 볼지.",
        )
        combo_ma_len = st.number_input(
            "이동평균 기간 (봉 수)", value=50, step=1,
            help="추세 방향을 판단할 때 쓸 이동평균선의 기간.",
        )
        combo_ma_type = st.selectbox(
            "이동평균 종류", ["EMA", "SMA", "WMA"], index=0,
            format_func=lambda v: {
                "EMA": "지수이동평균 (EMA)", "SMA": "단순이동평균 (SMA)", "WMA": "가중이동평균 (WMA)",
            }[v],
        )
        combo_score_threshold = st.number_input(
            "신호 점수 임계값 (0~1)", value=0.62, min_value=0.0, max_value=1.0,
            help="여러 조건을 종합한 점수가 이 값 이상일 때만 신호로 인정합니다. "
                 "원본 Pine Script가 채택한 값은 0.62이며, 이 대시보드로 직접 재검증하기 "
                 "전까지는 '일단 원본 그대로'라는 참고용 값으로 취급하는 게 안전합니다.",
        )

    run_clicked = st.button("검증 실행", type="primary", use_container_width=True)

if not run_clicked:
    st.info("왼쪽 사이드바에서 종목과 기간을 정하고 [검증 실행]을 눌러주세요. "
            "나머지 설정은 기본값 그대로 둬도 괜찮습니다.")
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
    st.warning("검증된 이벤트가 없습니다 (신호가 한 번도 발생하지 않았을 수 있습니다). "
               "기간을 늘리거나 종목을 바꿔서 다시 시도해보세요.")
    st.stop()

horizon_list = [int(h) for h in horizons.split(",")]
main_horizon = st.selectbox(
    "랭킹에 쓸 보유기간 (봉 수)", horizon_list, index=len(horizon_list) // 2,
    help="아래 표와 차트를 어느 보유기간 기준으로 볼지 고릅니다.",
)

sub = summary[summary["horizon"] == main_horizon].sort_values("t_stat", ascending=False)
st.subheader(f"보유기간 {main_horizon}봉 기준 t값 랭킹")
st.caption(
    "|t값| >= 2면 눈여겨볼 만한 신호, 표본수가 10개 미만이면 아직 표본이 부족합니다. "
    "'통계적 유의(BH보정)'는 이번 실행에서 나온 전체 신호 x 방향 x 보유기간 조합을 "
    "한꺼번에 보정한 뒤에도 유의하다고 판정된 것만 True입니다 — 개별 t값만 보고 "
    "판단하면 우연히 유의하게 나온 신호를 걸러내지 못할 수 있습니다."
)
with st.expander("표에 나오는 용어가 낯설다면"):
    st.markdown(
        "- **표본수**: 이 신호가 과거 데이터에서 몇 번 발생했는지. 적을수록(특히 10개 "
        "미만) 결과를 신뢰하기 어렵습니다.\n"
        "- **평균수익률 / 승률**: 신호 발생 후 해당 보유기간 동안의 평균 수익률과, "
        "수익이 났던 비율.\n"
        "- **t값**: 그 평균수익률이 우연(0%)과 통계적으로 얼마나 다른지를 나타내는 "
        "지표. 절댓값이 클수록 '우연이 아닐 가능성'이 높습니다. 보통 2 이상이면 "
        "눈여겨볼 만하다고 봅니다.\n"
        "- **p값**: t값을 '우연히 이런 결과가 나올 확률'로 환산한 값. 작을수록 "
        "우연이 아닐 가능성이 높습니다.\n"
        "- **통계적 유의 (BH보정)**: 여러 신호를 동시에 비교하면 그중 일부는 순전히 "
        "우연으로 유의하게 보일 수 있습니다. 이를 감안해 보정한 뒤에도 유의하다고 "
        "판정됐는지 여부입니다."
    )

display_sub = sub[["signal", "direction", "n_events", "mean_return_pct", "win_rate_pct",
                    "t_stat", "p_value", "significant_bh"]].reset_index(drop=True).copy()
display_sub["direction"] = display_sub["direction"].map(DIRECTION_LABELS).fillna(display_sub["direction"])
display_sub["significant_bh"] = display_sub["significant_bh"].map({True: "예", False: "아니오"})
display_sub = display_sub.rename(columns=COLUMN_LABELS)
st.dataframe(display_sub, use_container_width=True)

st.subheader("신호 발생 후 평균 누적수익률 경로")
st.caption("신호가 발생한 시점을 0으로 두고, 그 뒤 각 보유기간까지 평균적으로 수익률이 "
           "어떻게 누적되는지를 보여줍니다. (단위: %)")
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
        f"체결 로그 {log_path}에 {n_new}건 신규 기록 (보유기간={main_horizon}봉 기준). "
        f"청산 사유는 전부 '보유기간 만료'로 기록되며, 실제 손절/익절 로직이 적용된 건 "
        f"아닙니다."
    )
    if Path(log_path).exists():
        log_df = pd.read_csv(log_path)
        with st.expander(f"누적 체결 로그 미리보기 ({len(log_df)}행)"):
            st.dataframe(log_df.tail(50), use_container_width=True)
        st.download_button(
            "체결 로그 파일(execution_log.csv) 다운로드", data=Path(log_path).read_bytes(),
            file_name="execution_log.csv", mime="text/csv",
        )

st.caption(
    "주의: 이 화면은 과거 통계 기반 신호 요약이지 투자자문이 아닙니다. "
    "표본수가 작거나(특히 10개 미만) 통계적 유의성이 '아니오'면 결론으로 쓰기엔 이릅니다."
)
