#!/usr/bin/env python3
"""
alerts/watchlist.json에 적힌 종목들을 오늘자로 다시 돌려서, SIGNAL_SPECS에 등록된
신호 중 "마지막 봉(가장 최근 날짜)"에 새로 뜨고, 동시에 "이번 실행에서 BH(다중검정)
보정까지 통과한" 신호만 텔레그램으로 알린다.

[조건 강화] 단순히 이벤트 컬럼(crossover_confirmed 등)이 True라는 이유만으로는
알리지 않는다 — run_backtest.py가 매번 계산하는 summary의 significant_bh
(이번 실행에서 검증한 전체 신호x방향x보유기간 조합을 한꺼번에 놓고 Benjamini-
Hochberg 절차로 다중검정 보정한 뒤에도 유의한지)가 True인 신호만 통과시킨다.
어떤 보유기간(horizon) 기준으로 볼지는 run_backtest.main()이 랭킹을 보여줄 때
쓰는 것과 똑같은 "horizons 목록의 중간값" 관례를 그대로 따른다 — 알림 전용
기준을 새로 만들지 않고, 대시보드에서 보는 유의성 판단과 항상 같은 기준을 쓰기
위함.

[설계 원칙] run_backtest.py의 load_data()/run()을 그대로 재사용한다 — 신호 계산
로직을 이 파일에서 다시 구현하지 않는다(app.py와 동일한 '얇은 껍데기' 원칙). 즉
대시보드에서 보는 신호 정의와 알림이 쓰는 신호 정의는 항상 똑같다.

하루 한 번(GitHub Actions 스케줄)만 도는 "일일 다이제스트"다 — 장중 실시간 체결
알림이 아니다. 업비트(코인)는 24시간 거래되지만 이 스크립트가 도는 그 순간의
스냅샷만 보므로, 그 사이에 생겼다 사라진 신호는 못 잡는다.

사용법(저장소 루트에서): python -m alerts.check_signals
  [--watchlist alerts/watchlist.json] [--state alerts/state.json] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import run_backtest as rb
from backtest.engine import SIGNAL_SPECS

from alerts.config import args_for_watch_item, load_watchlist
from alerts.state import load_state, save_state
from alerts.telegram import send_telegram


def check_one(item: dict, state: dict) -> list[str]:
    """watchlist 항목 하나를 체크해서, 새로 보낼 알림 메시지 리스트를 반환하고
    state를 그 자리에서 갱신한다(아직 저장은 안 함 — 호출부가 전부 끝난 뒤 한 번에 저장)."""
    label = item.get("label", item["symbol"])
    args = args_for_watch_item(item)

    try:
        df = rb.load_data(args)
    except SystemExit as e:
        print(f"[건너뜀] {label}({item['symbol']}): 데이터 로딩 실패 - {e}")
        return []
    except Exception as e:
        print(f"[건너뜀] {label}({item['symbol']}): 데이터 로딩 중 오류 - {type(e).__name__}: {e}")
        return []

    if df is None or len(df) < 60:
        print(f"[건너뜀] {label}({item['symbol']}): 데이터가 부족합니다({0 if df is None else len(df)}봉).")
        return []

    summary, results, dfs_by_indicator, slippage_arg = rb.run(df, args)

    last_date = df.index[-1]
    last_close = df["close"].iloc[-1]
    messages = []

    # run_backtest.main()의 main_horizon과 동일한 규칙(horizons 목록의 중간값) —
    # 알림만을 위한 별도 기준을 새로 만들지 않는다.
    horizon_list = [int(h) for h in args.horizons.split(",")]
    judge_horizon = horizon_list[len(horizon_list) // 2]

    has_bh_col = "significant_bh" in summary.columns if summary is not None else False
    if not has_bh_col:
        print(f"[경고] {label}({item['symbol']}): summary에 significant_bh 컬럼이 "
              f"없습니다(표본이 너무 적어 BH보정 자체가 스킵됐을 수 있음) — 이번 "
              f"실행에서는 이 종목의 어떤 신호도 '조건 강화' 기준을 통과시키지 않습니다.")

    for indicator_key, event_col, direction, signal_label in SIGNAL_SPECS:
        df_i = dfs_by_indicator.get(indicator_key)
        if df_i is None or event_col not in df_i.columns:
            continue
        if not bool(df_i[event_col].iloc[-1]):
            continue

        if not has_bh_col:
            continue
        sig_row = summary[(summary["signal"] == signal_label) & (summary["horizon"] == judge_horizon)]
        bh_ok = bool(sig_row["significant_bh"].iloc[0]) if not sig_row.empty else False
        if not bh_ok:
            continue  # 원시 이벤트는 떴지만 BH보정을 통과 못한 신호 — 알리지 않음(조건 강화)

        state_key = f"{item['market']}:{item['symbol']}:{indicator_key}:{event_col}"
        date_str = str(last_date.date() if hasattr(last_date, "date") else last_date)
        if state.get(state_key) == date_str:
            continue  # 이미 같은 날짜로 알림을 보냈음 (중복 방지)

        direction_kr = "상승(롱)" if direction == 1 else "하락(숏)"
        messages.append(
            f"[신호 발생] {label} ({item['symbol']})\n"
            f"- 신호: {signal_label} ({direction_kr})\n"
            f"- 기준일: {date_str} / 종가: {last_close:,.2f}\n"
            f"- 보유기간 {judge_horizon}봉 기준, 이번 실행의 BH보정(다중검정) 통과 신호입니다.\n"
            f"- 참고: 그래도 과거 통계(t값·표본수)는 대시보드에서 직접 확인하세요. "
            f"이 알림은 '신호가 떴다'는 사실만 알려줄 뿐, 매매를 추천하는 것이 아닙니다."
        )
        state[state_key] = date_str

    return messages


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--watchlist", default="alerts/watchlist.json")
    p.add_argument("--state", default="alerts/state.json")
    p.add_argument("--dry-run", action="store_true",
                   help="텔레그램으로 실제 전송하지 않고 콘솔에만 출력한다(테스트용).")
    args = p.parse_args()

    watchlist = load_watchlist(args.watchlist)
    state = load_state(args.state)

    all_messages: list[str] = []
    for item in watchlist:
        all_messages.extend(check_one(item, state))

    if not all_messages:
        print("[결과] 새로 알릴 신호 없음.")
    else:
        print(f"[결과] 새 신호 {len(all_messages)}건.")
        for msg in all_messages:
            print("----")
            print(msg)
            if args.dry_run:
                continue
            try:
                send_telegram(msg)
            except Exception as e:
                print(f"[경고] 텔레그램 전송 실패: {type(e).__name__}: {e}", file=sys.stderr)

    save_state(args.state, state)
    print(f"[저장] {args.state}")


if __name__ == "__main__":
    main()
