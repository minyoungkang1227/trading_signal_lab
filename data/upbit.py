"""
업비트 공개 캔들 API에서 OHLCV 데이터를 받아와 표준 DataFrame으로 변환.
인증키 불필요 (공개 시세 조회 엔드포인트).

이 모듈의 fetch_* 함수는 data/collect.py에 정리된 수집 계약을 따른다:
반환 직전 attach_fetch_meta()로 fetched_at/source/symbol을 attrs에 남긴다.
"""
from __future__ import annotations

import time
import pandas as pd
import requests

from data.collect import attach_fetch_meta

UPBIT_BASE = "https://api.upbit.com/v1"


def _get(url: str, params: dict) -> list:
    r = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_upbit_ohlcv(market: str = "KRW-BTC", unit: str = "day", count: int = 500,
                       minute_unit: int = 60) -> pd.DataFrame:
    """
    market: 업비트 마켓 코드 (예: 'KRW-BTC', 'KRW-ETH')
    unit: 'day' | 'minute' | 'week'
    count: 가져올 캔들 개수 (200개씩 페이징해서 누적)
    minute_unit: unit='minute'일 때 분봉 단위 (1,3,5,15,30,60,240)

    반환: index=날짜(UTC), columns=[open, high, low, close, volume] 오름차순 정렬 DataFrame
    """
    if unit == "day":
        path = f"{UPBIT_BASE}/candles/days"
    elif unit == "week":
        path = f"{UPBIT_BASE}/candles/weeks"
    elif unit == "minute":
        path = f"{UPBIT_BASE}/candles/minutes/{minute_unit}"
    else:
        raise ValueError("unit must be 'day', 'week', or 'minute'")

    rows = []
    to = None
    remaining = count
    while remaining > 0:
        batch = min(200, remaining)
        params = {"market": market, "count": batch}
        if to:
            params["to"] = to
        data = _get(path, params)
        if not data:
            break
        rows.extend(data)
        to = data[-1]["candle_date_time_utc"]
        remaining -= len(data)
        if len(data) < batch:
            break
        time.sleep(0.12)  # 업비트 레이트리밋 여유

    if not rows:
        raise RuntimeError(f"업비트에서 {market} 캔들을 가져오지 못했습니다.")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["candle_date_time_utc"])
    df = df.rename(columns={
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume",
    })
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return attach_fetch_meta(df, symbol=market, source="upbit")


def fetch_upbit_ohlcv_range(market: str, start, end, unit: str = "day") -> pd.DataFrame:
    """
    fetch_upbit_ohlcv()는 "최근 N개 캔들"(count) 기준이라, 다른 수집 함수들
    (fetch_krx_ohlcv 등)과 인터페이스가 심볼+기간으로 통일되지 않는 문제가
    있었다. 이 함수는 동일한 페이징 로직을 쓰되 start/end 날짜로 필요한
    캔들 개수를 역산해서, "심볼과 기간만 입력받는다"는 수집 계약을 맞춘다.

    start, end: 'YYYYMMDD' 또는 date-like. unit: 'day' | 'week' | 'minute'(분 단위는 60분 고정)
    """
    start_ts = pd.to_datetime(start, format="%Y%m%d" if isinstance(start, str) and len(start) == 8 else None)
    end_ts = pd.to_datetime(end, format="%Y%m%d" if isinstance(end, str) and len(end) == 8 else None)
    if unit == "day":
        n_days = max((end_ts - start_ts).days + 5, 1)
    elif unit == "week":
        n_days = max((end_ts - start_ts).days // 7 + 3, 1)
    else:
        n_days = max((end_ts - start_ts).days * 24 + 10, 1)

    df = fetch_upbit_ohlcv(market=market, unit=unit, count=n_days)
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    return attach_fetch_meta(df, symbol=market, source="upbit")
