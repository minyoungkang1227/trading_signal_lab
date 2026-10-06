"""
국내 주식 일봉 OHLCV 수집. 1차로 pykrx(KRX 정식 데이터)를 쓰고, 설치가
안 됐거나 실패하면 yfinance(티커.KS)로 대체한다. KRX Open API(정식 발급
AUTH_KEY 필요, data.krx.co.kr)를 쓰는 fetch_krx_open_api_ohlcv()도 별도로
제공한다.

이 모듈의 모든 fetch_* 함수는 data/collect.py에 정리된 수집 계약을 따른다:
심볼+기간만 입력받아 표준 컬럼의 DataFrame을 반환하고, 반환 직전
attach_fetch_meta()로 fetched_at/source/symbol을 attrs에 남긴다.
"""
from __future__ import annotations

import pandas as pd

from data.collect import attach_fetch_meta, assert_unique_columns


def fetch_krx_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    ticker: 6자리 종목코드 (예: '005930' 삼성전자, '000660' SK하이닉스)
    start, end: 'YYYYMMDD' 형식
    반환: index=날짜, columns=[open, high, low, close, volume] 오름차순 DataFrame
    """
    try:
        from pykrx import stock
        df = stock.get_market_ohlcv(start, end, ticker)
        df = df.rename(columns={
            "시가": "open", "고가": "high", "저가": "low",
            "종가": "close", "거래량": "volume",
        })
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        df = assert_unique_columns(df, context=f"krx/pykrx:{ticker}")
        df.index.name = "date"
        return attach_fetch_meta(df, symbol=ticker, source="pykrx")
    except Exception as e:
        print(f"[krx] pykrx 실패({e}), yfinance로 재시도합니다.")

    import yfinance as yf
    yf_ticker = f"{ticker}.KS"
    df = yf.download(yf_ticker, start=pd.to_datetime(start, format="%Y%m%d"),
                      end=pd.to_datetime(end, format="%Y%m%d"), progress=False)
    if df.empty:
        raise RuntimeError(f"{ticker} 데이터를 pykrx/yfinance 둘 다에서 가져오지 못했습니다.")
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                             "Close": "close", "Volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    df = assert_unique_columns(df, context=f"krx/yfinance:{ticker}")
    df.index.name = "date"
    return attach_fetch_meta(df, symbol=ticker, source="yfinance")


# KRX Open API(data.krx.co.kr, "유가증권 일별매매정보" stk_bydd_trd) 응답
# 필드명. 이 엔드포인트는 "기준일자(basDd) 하나 -> 그날 전종목 스냅샷"
# 형태라서, 기간 조회는 영업일마다 반복 호출 후 종목코드로 필터링해야 한다.
# 아래 필드명은 KRX Open API의 통상적 네이밍 관례를 따른 것으로, 이
# 샌드박스는 외부 네트워크가 차단돼 있어 실제 응답으로 검증하지 못했다 —
# 처음 실행할 때 KeyError가 나면 이 딕셔너리만 실제 응답 필드명으로
# 고치면 된다(호출부 로직은 그대로 두면 됨).
_KRX_OPEN_API_FIELDS = {
    "code": "ISU_SRT_CD",
    "open": "TDD_OPNPRC",
    "high": "TDD_HGPRC",
    "low": "TDD_LWPRC",
    "close": "TDD_CLSPRC",
    "volume": "ACC_TRDVOL",
}
_KRX_OPEN_API_URL = "https://data.krx.co.kr/svc/apis/sto/stk_bydd_trd"


def fetch_krx_open_api_ohlcv(ticker: str, start: str, end: str, auth_key: str | None = None) -> pd.DataFrame:
    """
    새로 발급받은 KRX Open API(AUTH_KEY 방식)로 일봉 OHLCV를 수집한다.
    pykrx보다 이 샌드박스 밖(실제 네트워크가 열린 환경)에서 더 안정적으로
    동작할 것으로 기대하고 추가한 대체 경로다.

    ticker: 6자리 종목코드
    start, end: 'YYYYMMDD'
    auth_key: KRX Open API 발급키. None이면 환경변수 KRX_OPEN_API_KEY를 사용.

    주의: stk_bydd_trd 엔드포인트는 종목별 기간조회가 아니라 "기준일자 1개
    -> 전종목 스냅샷"을 반환하는 구조로 알려져 있어, 영업일 수만큼 API를
    반복 호출한다(기간이 길면 느릴 수 있음). 응답 필드명은
    _KRX_OPEN_API_FIELDS에서 한 곳에 모아뒀다 — 실제 응답과 다르면 여기만
    고치면 된다.
    """
    import os
    import requests

    auth_key = auth_key or os.environ.get("KRX_OPEN_API_KEY")
    if not auth_key:
        raise RuntimeError("KRX Open API 인증키가 없습니다. auth_key 인자나 "
                            "환경변수 KRX_OPEN_API_KEY를 설정하세요.")

    business_days = pd.bdate_range(pd.to_datetime(start, format="%Y%m%d"),
                                    pd.to_datetime(end, format="%Y%m%d"))
    rows = []
    f = _KRX_OPEN_API_FIELDS
    for d in business_days:
        bas_dd = d.strftime("%Y%m%d")
        resp = requests.get(_KRX_OPEN_API_URL, headers={"AUTH_KEY": auth_key},
                             params={"basDd": bas_dd}, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("OutBlock_1", payload.get("output", []))
        for item in items:
            if item.get(f["code"]) != ticker:
                continue
            rows.append({
                "date": d,
                "open": float(str(item[f["open"]]).replace(",", "")),
                "high": float(str(item[f["high"]]).replace(",", "")),
                "low": float(str(item[f["low"]]).replace(",", "")),
                "close": float(str(item[f["close"]]).replace(",", "")),
                "volume": float(str(item[f["volume"]]).replace(",", "")),
            })
            break

    if not rows:
        raise RuntimeError(f"KRX Open API에서 {ticker}의 {start}~{end} 데이터를 찾지 못했습니다.")

    df = pd.DataFrame(rows).set_index("date").sort_index()
    return attach_fetch_meta(df, symbol=ticker, source="krx_open_api")
