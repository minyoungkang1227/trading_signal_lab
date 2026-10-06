#!/usr/bin/env python3
"""
한국시장(KRX) SMB·HML 팩터를 매월 말 리밸런싱으로 계산해 csv로 저장하는 CLI.

pykrx로 매 리밸런싱 시점마다 KOSPI+KOSDAQ 전종목 시가총액·PBR을 조회하므로
호출 횟수가 많고 느리다(기간이 길수록 오래 걸림). 샌드박스처럼 외부 네트워크가
막힌 환경에서는 동작하지 않으니, 로컬 PC나 서버에서 실행할 것.

사용 예:
  python build_kr_factors.py --start 20220101 --end 20240101 --output output/kr_factors.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from factors.kr_fama_french import month_end_dates, compute_smb_hml


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="YYYYMMDD")
    p.add_argument("--end", required=True, help="YYYYMMDD")
    p.add_argument("--output", default="output/kr_factors.csv")
    p.add_argument("--delisted-return", dest="delisted_return", type=float, default=None,
                    help="리밸런싱 구간 중 상장폐지·거래정지 등으로 종가 조회가 끊긴 종목의 "
                         "처리 방식. 기본 None=조용히 제외(생존 편향 발생 가능). -1.0을 주면 "
                         "그런 종목을 전손으로 취급해 편향을 줄인다(pykrx가 상장폐지 종목의 "
                         "과거 스냅샷을 이미 빼고 반환하는 경우, 이 옵션은 '조회는 됐지만 다음 "
                         "시점에 사라진' 종목만 잡아낸다는 한계가 있다).")
    args = p.parse_args()

    dates = month_end_dates(args.start, args.end)
    print(f"[리밸런싱 시점] {len(dates)}개 월말 날짜 ({dates[0]} ~ {dates[-1]})")

    factors = compute_smb_hml(dates, delisted_return=args.delisted_return)
    if factors.empty:
        print("[경고] 계산된 팩터가 없습니다. 날짜가 전부 비영업일이었을 수 있습니다.")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(out_path, encoding="utf-8-sig")
    print(f"[저장] {out_path}")
    print(factors.describe())


if __name__ == "__main__":
    main()
