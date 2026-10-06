"""
한국시장(KRX) 데이터로 Fama & French(1993) SMB·HML 팩터를 구성한다.

원 논문 방법론
--------------
1. 매년 6월 말, NYSE 상장 종목의 시가총액(ME) 중앙값을 기준으로 전체 유니버스를
   Small(S)/Big(B) 둘로 나눈다. NYSE 종목만으로 breakpoint를 잡는 이유는,
   NASDAQ에 극단적으로 작은 소형주가 많이 몰려 있어서 전체 유니버스로 breakpoint를
   잡으면 Small 그룹에 종목이 지나치게 쏠리기 때문이다.
2. 마찬가지로 NYSE 종목만으로 장부가/시가(BE/ME) 30%, 70% breakpoint를 잡아
   전체 유니버스를 Low(L, 성장주)/Medium(M)/High(H, 가치주) 셋으로 나눈다.
   BE는 전년도 12월 결산 장부가치, ME는 그 시점 시가총액이다.
3. 사이즈 2개 × BE/ME 3개 = 6개 포트폴리오를 시가총액가중으로 구성하고,
   SMB = (S/L+S/M+S/H)/3 - (B/L+B/M+B/H)/3
   HML = (S/H+B/H)/2 - (S/L+B/L)/2
   로 두 개의 롱숏 요인 포트폴리오를 만든다. S와 B 양쪽에 L/M/H를 고르게
   섞었기 때문에 SMB에서는 가치효과가, HML에서는 사이즈효과가 서로 상쇄된다.

한국시장에 대응시킨 부분 (근사)
--------------------------------
- NYSE ≈ KOSPI, NASDAQ ≈ KOSDAQ 대응관계를 차용했다. KOSDAQ에 소형·성장주가
  상대적으로 많이 몰려 있다는 구조가 NASDAQ과 유사하다고 보고, 사이즈·BE/ME
  breakpoint를 KOSPI 종목만으로 계산한다(assign_groups).
- BE/ME는 직접 재무제표(자본총계)를 가져오는 대신, pykrx가 매 영업일 제공하는
  PBR(=시가총액/자본총계)의 역수로 근사한다: BE/ME = 1/PBR.
- 원 논문은 연 1회(6월 말) 리밸런싱하며 전년도 12월 결산 데이터를 쓰지만,
  여기서는 매월 말 리밸런싱으로 단순화했다(month_end_dates). PBR이 최근 자본
  총계를 이미 반영한 값이라, 원 논문보다 정보가 더 빨리(그리고 더 자주) 갱신된
  결과가 나온다는 점을 감안해서 해석해야 한다.
- 완전자본잠식 등으로 PBR<=0인 종목은 BE/ME 정의가 성립하지 않아 유니버스에서
  제외한다.

이 모듈은 pykrx로 실시간 KRX 데이터를 받아오므로, 외부 네트워크가 막힌 샌드박스
환경에서는 동작하지 않는다. `_smoke_test_factors.py`는 fetch_fn을 가짜 함수로
주입해 네트워크 없이 계산 로직(그룹핑·집계식)만 검증한다.
"""
from __future__ import annotations
import pandas as pd


def fetch_cross_section(date: str) -> pd.DataFrame:
    """
    date: 'YYYYMMDD' (영업일이어야 한다).
    그날 KOSPI+KOSDAQ 전종목의 시가총액·종가·BE/ME를 모은 횡단면 데이터를
    ticker를 인덱스로 하는 DataFrame으로 반환한다.
    컬럼: market('KOSPI'/'KOSDAQ'), close(종가), mktcap(시가총액), be_me(1/PBR)
    """
    from pykrx import stock

    frames = []
    for market in ("KOSPI", "KOSDAQ"):
        cap = stock.get_market_cap_by_ticker(date, market=market)
        fund = stock.get_market_fundamental_by_ticker(date, market=market)
        df = cap.join(fund[["PBR"]], how="inner")
        df["market"] = market
        frames.append(df)

    out = pd.concat(frames)
    out = out.rename(columns={"종가": "close", "시가총액": "mktcap", "PBR": "pbr"})
    out = out[(out["mktcap"] > 0) & (out["pbr"] > 0)]  # PBR<=0은 BE/ME 정의 불가
    out["be_me"] = 1.0 / out["pbr"]
    out.index.name = "ticker"
    return out[["market", "close", "mktcap", "be_me"]]


def assign_groups(cross_section: pd.DataFrame, bm_breakpoints: tuple[float, float] = (0.3, 0.7)) -> pd.DataFrame:
    """
    KOSPI 종목만으로 사이즈 중앙값, BE/ME 30/70 percentile을 계산해
    전체 종목(KOSPI+KOSDAQ)에 적용한다. 'size_grp'(S/B), 'bm_grp'(L/M/H) 컬럼 추가.
    """
    out = cross_section.copy()
    kospi = out[out["market"] == "KOSPI"]
    if kospi.empty:
        raise ValueError("KOSPI 종목이 없어 breakpoint를 계산할 수 없습니다 (cross_section 확인).")

    size_bp = kospi["mktcap"].median()
    out["size_grp"] = out["mktcap"].apply(lambda x: "B" if x >= size_bp else "S")

    lo = kospi["be_me"].quantile(bm_breakpoints[0])
    hi = kospi["be_me"].quantile(bm_breakpoints[1])

    def _bm_grp(x: float) -> str:
        if x <= lo:
            return "L"
        if x >= hi:
            return "H"
        return "M"

    out["bm_grp"] = out["be_me"].apply(_bm_grp)
    return out


def _value_weighted_return(weight_by_ticker: pd.Series, close_t0: pd.Series, close_t1: pd.Series,
                            delisted_return: float | None = None) -> float:
    """
    weight_by_ticker: 포트폴리오에 속한 종목의 시가총액(가중치로 씀, 합계는 내부에서 정규화).
    close_t0/close_t1: 전체 유니버스의 t0/t1 시점 종가 (ticker -> 종가).
    delisted_return: t0에는 있었지만 t1에는 없는 종목(구간 중 상장폐지·거래정지 등으로
        종가 조회가 안 되는 경우)을 어떻게 처리할지.
          - None(기본값, 이전 동작과 동일): 그런 종목을 포트폴리오에서 조용히 제외하고,
            남은 종목들끼리만 가중치를 재정규화한다. 상장폐지는 거의 항상 큰 손실로
            끝나므로, 이렇게 빼면 포트폴리오 수익률이 실제보다 좋게(위로) 편향된다
            — 생존 편향(survivorship bias).
          - 숫자(예: -1.0 = 전손): t0 시점 가중치는 그대로 유지한 채, t1에서 사라진
            종목에 이 수익률을 강제로 대입한다. -1.0이면 상장폐지 종목을 투자금 전액
            손실로 취급하는 가장 보수적인 근사다.
    반환: 시가총액가중 포트폴리오 수익률. t0에 있는 종목이 하나도 없으면 NaN.
    """
    have_t0 = weight_by_ticker.index.intersection(close_t0.index)
    if len(have_t0) == 0:
        return float("nan")

    present_t1 = have_t0.intersection(close_t1.index)
    missing_t1 = have_t0.difference(close_t1.index)

    if delisted_return is None:
        # 이전 동작: t1에 없는 종목은 그냥 빼고, 남은 종목끼리만 가중치 재정규화.
        if len(present_t1) == 0:
            return float("nan")
        w = weight_by_ticker.loc[present_t1]
        w = w / w.sum()
        ret = close_t1.loc[present_t1] / close_t0.loc[present_t1] - 1
        return float((w * ret).sum())

    # delisted_return 지정: t0 시점 가중치 비중은 그대로 유지(재정규화는 have_t0 전체
    # 기준으로만 한 번), 사라진 종목에는 delisted_return을 강제로 대입.
    w = weight_by_ticker.loc[have_t0]
    w = w / w.sum()
    ret = pd.Series(index=have_t0, dtype=float)
    if len(present_t1) > 0:
        ret.loc[present_t1] = close_t1.loc[present_t1] / close_t0.loc[present_t1] - 1
    if len(missing_t1) > 0:
        ret.loc[missing_t1] = delisted_return
    return float((w * ret).sum())


def month_end_dates(start: str, end: str) -> list[str]:
    """
    start, end: 'YYYYMMDD'. 달력상 매월 말일을 'YYYYMMDD' 문자열 리스트로 반환한다
    (오름차순). 주말/공휴일이라 실제 영업일이 아닐 수 있으므로, compute_smb_hml은
    fetch_fn 호출이 실패하면 해당 구간을 건너뛰고 경고를 출력한다 — 엄밀하게
    맞추려면 이 함수가 반환한 날짜를 pykrx의 영업일 캘린더로 스냅한 뒤 넘길 것.
    """
    idx = pd.date_range(start, end, freq="ME")
    return [d.strftime("%Y%m%d") for d in idx]


def compute_smb_hml(rebalance_dates: list[str], fetch_fn=fetch_cross_section,
                     delisted_return: float | None = None) -> pd.DataFrame:
    """
    rebalance_dates: 리밸런싱 시점('YYYYMMDD') 리스트, 오름차순 (예: month_end_dates 결과).
    각 구간 [t0, t1]에 대해 t0 시점 그룹핑(사이즈·BE/ME)으로 6개 포트폴리오를 만들고,
    t0->t1 구간의 시가총액가중 수익률로 SMB, HML을 계산한다.

    fetch_fn: date -> DataFrame(index=ticker, columns=[market, close, mktcap, be_me]).
              기본값은 실제 pykrx 조회(fetch_cross_section). 테스트 시 가짜 함수를
              주입해 네트워크 없이 집계 로직만 검증할 수 있다.
    delisted_return: _value_weighted_return()에 그대로 전달. 기본 None은 이전 동작과
              동일(상장폐지 종목 조용히 제외 -> 생존 편향). 예: -1.0을 주면 t0~t1
              구간 중 종가 조회가 끊긴(상장폐지·장기 거래정지 추정) 종목을 전손으로
              취급해 포트폴리오 수익률에 반영한다.

    반환: index=t1 날짜('YYYYMMDD'), columns=[SMB, HML, n_stocks]
    """
    rows = []
    for t0, t1 in zip(rebalance_dates[:-1], rebalance_dates[1:]):
        try:
            cs0 = assign_groups(fetch_fn(t0))
            close_t1 = fetch_fn(t1)["close"]
        except Exception as e:
            print(f"[경고] {t0}~{t1} 구간 데이터 조회 실패, 건너뜁니다: {e}")
            continue

        port_returns = {}
        for size_grp in ("S", "B"):
            for bm_grp in ("L", "M", "H"):
                bucket = cs0[(cs0["size_grp"] == size_grp) & (cs0["bm_grp"] == bm_grp)]
                port_returns[(size_grp, bm_grp)] = _value_weighted_return(
                    bucket["mktcap"], cs0["close"], close_t1, delisted_return=delisted_return
                )

        smb = (sum(port_returns[("S", g)] for g in "LMH") / 3
               - sum(port_returns[("B", g)] for g in "LMH") / 3)
        hml = ((port_returns[("S", "H")] + port_returns[("B", "H")]) / 2
               - (port_returns[("S", "L")] + port_returns[("B", "L")]) / 2)

        rows.append(dict(date=t1, SMB=smb, HML=hml, n_stocks=len(cs0)))

    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["SMB", "HML", "n_stocks"])


# ---------------------------------------------------------------------------
# Carhart(1997) WML(Winners Minus Losers, 원 논문 표기 UMD) 팩터
# ---------------------------------------------------------------------------
"""
WML 구성 방법 (SMB/HML과 동일한 뼈대에 정렬 기준만 하나 추가)
----------------------------------------------------------
1. 사이즈(S/B)는 assign_groups()의 size_grp를 그대로 재사용한다.
2. 두 번째 정렬 기준을 BE/ME 대신 "형성기간 모멘텀 수익률"로 바꾼다. 원 논문은
   리밸런싱 시점 t로부터 t-12개월~t-2개월 구간의 누적수익률을 쓴다 — 가장 최근
   1개월(t-1~t)은 일부러 뺀다. 이유는 그 구간에 단기반전(short-term reversal,
   호가 바운스 등 미시구조 효과로 최근 1개월 수익률이 다음 기간에 반대로 튀는
   현상)이 섞여 모멘텀 신호를 오염시키기 때문이다.
3. 이 모멘텀 수익률로 KOSPI 종목만 30/70 percentile breakpoint를 잡아
   Loser(L)/Middle(M)/Winner(W) 셋으로 나누고, 사이즈 2개 × 모멘텀 3개 = 6개
   포트폴리오를 시가총액가중으로 구성한다.
4. WML = (S/W+B/W)/2 - (S/L+B/L)/2 — 승자 매수, 패자 매도 롱숏 포트폴리오.
   SMB/HML을 만들 때와 마찬가지로 S/B 양쪽에 W/L을 고르게 섞어 사이즈효과를
   상쇄시켰다.

주의: 모멘텀 수익률을 계산하려면 리밸런싱 시점 이전의 과거 가격이 추가로
필요하다. fetch_cross_section()은 그 시점 하나만의 스냅샷이라 이 정보가 없으므로,
과거 두 시점(t-12개월, t-2개월)의 종가 횡단면을 따로 조회하는
fetch_momentum_cross_section()을 새로 둔다.
"""


def fetch_close_cross_section(date: str) -> pd.Series:
    """
    date: 'YYYYMMDD' (영업일).
    그날 KOSPI+KOSDAQ 전종목의 종가 스냅샷을 ticker->종가 Series로 반환한다.
    fetch_cross_section과 달리 시가총액·PBR 없이 종가만 필요할 때(모멘텀 계산용) 쓴다.
    """
    from pykrx import stock

    closes = []
    for market in ("KOSPI", "KOSDAQ"):
        df = stock.get_market_ohlcv_by_ticker(date, market=market)
        closes.append(df["종가"])
    out = pd.concat(closes)
    out.index.name = "ticker"
    out.name = "close"
    return out


def fetch_momentum_cross_section(formation_date: str, lookback_months: int = 12,
                                  skip_months: int = 1) -> pd.Series:
    """
    formation_date: 리밸런싱 시점('YYYYMMDD').
    t-lookback_months월 ~ t-skip_months월 구간의 누적수익률을 종목별로 계산한다
    (Carhart 원 논문 기본값: lookback=12, skip=1, 즉 t-12개월~t-1개월 시점 종가 비교).
    반환: ticker -> 모멘텀 수익률 Series. 두 시점 모두에 가격이 있는 종목만 포함.
    """
    t0 = (pd.Timestamp(formation_date) - pd.DateOffset(months=lookback_months)).strftime("%Y%m%d")
    t1 = (pd.Timestamp(formation_date) - pd.DateOffset(months=skip_months)).strftime("%Y%m%d")
    close_t0 = fetch_close_cross_section(t0)
    close_t1 = fetch_close_cross_section(t1)
    common = close_t0.index.intersection(close_t1.index)
    mom = (close_t1.loc[common] / close_t0.loc[common] - 1)
    mom.name = "mom_return"
    return mom


def assign_momentum_groups(cross_section: pd.DataFrame, mom_return: pd.Series,
                            mom_breakpoints: tuple[float, float] = (0.3, 0.7)) -> pd.DataFrame:
    """
    cross_section: assign_groups()를 거쳐 size_grp가 이미 있는 DataFrame.
    mom_return: fetch_momentum_cross_section() 반환값(ticker -> 모멘텀 수익률).
    KOSPI 종목만으로 모멘텀 30/70 percentile breakpoint를 잡아 전체 종목에 적용한다.
    'mom_grp'(L/M/W) 컬럼 추가. mom_return이 없는 종목(신규상장 등)은 제외한다.
    """
    out = cross_section.copy()
    out["mom_return"] = mom_return.reindex(out.index)
    out = out.dropna(subset=["mom_return"])

    kospi = out[out["market"] == "KOSPI"]
    if kospi.empty:
        raise ValueError("KOSPI 종목이 없어 모멘텀 breakpoint를 계산할 수 없습니다.")

    lo = kospi["mom_return"].quantile(mom_breakpoints[0])
    hi = kospi["mom_return"].quantile(mom_breakpoints[1])

    def _mom_grp(x: float) -> str:
        if x <= lo:
            return "L"
        if x >= hi:
            return "W"
        return "M"

    out["mom_grp"] = out["mom_return"].apply(_mom_grp)
    return out


def compute_wml(rebalance_dates: list[str], fetch_fn=fetch_cross_section,
                 fetch_mom_fn=fetch_momentum_cross_section,
                 delisted_return: float | None = None) -> pd.DataFrame:
    """
    rebalance_dates: 리밸런싱 시점('YYYYMMDD') 리스트, 오름차순.
    각 구간 [t0, t1]에 대해 t0 시점 사이즈×모멘텀 그룹핑으로 6개 포트폴리오를 만들고,
    t0->t1 구간의 시가총액가중 수익률로 WML을 계산한다.

    fetch_fn: compute_smb_hml과 동일 — date -> 시가총액·종가 등 횡단면 DataFrame.
    fetch_mom_fn: date -> 모멘텀 수익률 Series (fetch_momentum_cross_section 기본값).
                  둘 다 테스트 시 가짜 함수로 교체해 네트워크 없이 검증할 수 있다.
    delisted_return: compute_smb_hml과 동일한 의미로 _value_weighted_return()에 전달.
                  기본 None은 이전 동작(상장폐지 종목 제외)과 동일.

    반환: index=t1 날짜('YYYYMMDD'), columns=[WML, n_stocks]
    """
    rows = []
    for t0, t1 in zip(rebalance_dates[:-1], rebalance_dates[1:]):
        try:
            cs0 = assign_groups(fetch_fn(t0))
            cs0 = assign_momentum_groups(cs0, fetch_mom_fn(t0))
            close_t1 = fetch_fn(t1)["close"]
        except Exception as e:
            print(f"[경고] {t0}~{t1} 구간 데이터 조회 실패, 건너뜁니다: {e}")
            continue

        port_returns = {}
        for size_grp in ("S", "B"):
            for mom_grp in ("L", "M", "W"):
                bucket = cs0[(cs0["size_grp"] == size_grp) & (cs0["mom_grp"] == mom_grp)]
                port_returns[(size_grp, mom_grp)] = _value_weighted_return(
                    bucket["mktcap"], cs0["close"], close_t1, delisted_return=delisted_return
                )

        wml = ((port_returns[("S", "W")] + port_returns[("B", "W")]) / 2
               - (port_returns[("S", "L")] + port_returns[("B", "L")]) / 2)

        rows.append(dict(date=t1, WML=wml, n_stocks=len(cs0)))

    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["WML", "n_stocks"])


def combine_factors(smb_hml_df: pd.DataFrame, wml_df: pd.DataFrame) -> pd.DataFrame:
    """
    compute_smb_hml()과 compute_wml() 결과를 날짜 기준으로 합쳐 SMB/HML/WML
    세 컬럼을 가진 하나의 DataFrame으로 만든다. backtest.engine.factor_adjust()가
    받는 factor_df 형태가 바로 이거다.

    두 함수를 같은 rebalance_dates로 호출했다면 날짜가 거의 겹치지만, 한쪽만
    데이터 조회에 실패해 날짜가 빠진 경우를 대비해 outer join한다(결측은 NaN으로
    남고, factor_adjust는 그 NaN 행을 회귀에서 자동으로 제외한다).
    """
    out = smb_hml_df[["SMB", "HML"]].join(wml_df[["WML"]], how="outer")
    out.index.name = "date"
    return out
