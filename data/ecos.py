"""
한국은행 ECOS(경제통계시스템) Open API에서 거시 시계열(원달러 환율,
기준금리 등)을 받아온다. KRX/Upbit 시세 데이터를 검증(cross validation)하는
용도 — 예: 원달러 환율을 ECOS와 FinanceDataReader 양쪽에서 받아 같은
날짜에 값이 일치하는지 확인.

이 모듈의 fetch_* 함수도 data/collect.py의 수집 계약을 따른다: 반환 직전
attach_fetch_meta()로 fetched_at/source/symbol(=stat_code:item_code1)을
attrs에 남긴다.

ECOS REST URL 형식(공식 문서 기준):
  https://ecos.bok.or.kr/api/StatisticSearch/{인증키}/json/kr/
    {요청시작건수}/{요청종료건수}/{통계표코드}/{주기}/
    {검색시작일자}/{검색종료일자}/{통계항목코드1}

주기(cycle)는 A(연)/S(반년)/Q(분기)/M(월)/SM(반월)/D(일) 중 하나이며,
검색시작/종료일자의 형식은 주기에 따라 달라진다(D는 YYYYMMDD, M은
YYYYMM 등). 이 샌드박스는 외부 네트워크가 차단돼 있어 실제 응답으로
검증하지 못했으므로, 처음 실행할 때 401/오류가 나면 인증키·통계표코드·
날짜형식을 ECOS 오픈API 활용가이드에서 다시 확인할 것.
"""
from __future__ import annotations

import os

import pandas as pd

from data.collect import attach_fetch_meta

ECOS_BASE = "https://ecos.bok.or.kr/api/StatisticSearch"

# 자주 쓰는 통계표코드 예시(ECOS 통계코드 검색에서 확인한 관례적 코드 —
# 실제 호출 전 https://ecos.bok.or.kr 의 "통계코드검색"으로 재확인 권장).
STAT_CODES = {
    "usdkrw": ("731Y001", "0000001"),   # 시장평균환율(원/달러)
    "base_rate": ("722Y001", "0101000"),  # 한국은행 기준금리
}


def fetch_ecos_series(stat_code: str, item_code1: str, start: str, end: str,
                       cycle: str = "D", auth_key: str | None = None) -> pd.Series:
    """
    stat_code, item_code1: ECOS 통계표코드/통계항목코드1
    start, end: cycle에 맞는 날짜 형식(cycle='D'면 'YYYYMMDD', 'M'이면 'YYYYMM' 등)
    cycle: 'A'|'S'|'Q'|'M'|'SM'|'D'
    auth_key: None이면 환경변수 ECOS_API_KEY 사용

    반환: index=date, name=stat_code, 값=float인 pd.Series (attrs에 수집 메타데이터 포함)
    """
    import requests

    auth_key = auth_key or os.environ.get("ECOS_API_KEY")
    if not auth_key:
        raise RuntimeError("ECOS API 인증키가 없습니다. auth_key 인자나 "
                            "환경변수 ECOS_API_KEY를 설정하세요.")

    url = (f"{ECOS_BASE}/{auth_key}/json/kr/1/10000/"
           f"{stat_code}/{cycle}/{start}/{end}/{item_code1}")
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    payload = resp.json()

    if "RESULT" in payload:
        raise RuntimeError(f"ECOS API 오류: {payload['RESULT']}")
    rows = payload.get("StatisticSearch", {}).get("row", [])
    if not rows:
        raise RuntimeError(f"ECOS에서 {stat_code}/{item_code1} 데이터를 찾지 못했습니다.")

    dates = [r["TIME"] for r in rows]
    values = [float(r["DATA_VALUE"]) for r in rows]
    date_fmt = {"D": "%Y%m%d", "M": "%Y%m", "Q": None, "A": "%Y"}.get(cycle)
    idx = pd.to_datetime(dates, format=date_fmt) if date_fmt else pd.to_datetime(dates)

    s = pd.Series(values, index=idx, name=stat_code).sort_index()
    return attach_fetch_meta(s, symbol=f"{stat_code}:{item_code1}", source="ecos")


def fetch_usdkrw(start: str, end: str, auth_key: str | None = None) -> pd.Series:
    """자주 쓰는 원달러 환율 조회를 위한 편의 함수."""
    stat_code, item_code1 = STAT_CODES["usdkrw"]
    return fetch_ecos_series(stat_code, item_code1, start, end, cycle="D", auth_key=auth_key)
