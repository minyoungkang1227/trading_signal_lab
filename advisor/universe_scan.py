"""
여러 종목(유니버스)을 한 번에 훑어서 두 가지를 만든다.

  1. compute_pooled_stats() — 신호 유형별(예: "델타트레이딩-골든크로스") 횡단면
     풀링 통계. evaluator.evaluate_symbol()이 개별 종목의 표본이 부족할 때
     기대는 하이브리드 신뢰도 판단의 재료.
  2. scan_universe()        — 유니버스 전체에 evaluator.evaluate_symbol()을
     돌려서 "지금 매수/매도 신호가 뜬 종목" 랭킹표를 만든다. run_advisor.py의
     --mode scan이 이 결과를 캐시 파일(output/advisor_scan.csv,
     output/advisor_pooled_stats.csv)로 저장해두면, --mode query는 매번
     전체 유니버스를 다시 훑지 않고 이 캐시에서 "다른 유망 종목"을 뽑아 쓴다
     (실시간 전체 스캔은 종목 수가 많아지면(KRX 300 등) 너무 느리고 API
     호출량도 커지기 때문 — 배치 캐싱).
"""
from __future__ import annotations

import pandas as pd

from backtest.engine import SIGNAL_SPECS, event_returns, cluster_robust_t_stat
from research.param_search import pool_event_returns
from advisor.evaluator import evaluate_symbol, _compute_indicator_dfs


def compute_pooled_stats(dfs_by_symbol: dict[str, pd.DataFrame], cfg) -> dict[str, dict]:
    """
    dfs_by_symbol: {symbol: OHLCV df, ...} (유니버스 전체).
    cfg: evaluator._compute_indicator_dfs와 동일한 설정 객체.

    반환: {signal_label: {"n": int, "mean": float, "t_stat": float}}. n=0이면
    그 신호가 유니버스 어디에서도 한 번도 뜨지 않았다는 뜻(풀링 불가).

    cluster_robust_t_stat()으로 계산한다 — 여러 종목을 풀링하면 "같은 날 여러
    종목에서 동시에 이벤트가 뜨는" 횡단면 상관이 생기는데, 이걸 무시하면
    t-stat이 부풀려진다(research/param_search.py, 8절 참고).
    """
    indicator_dfs_by_symbol = {symbol: _compute_indicator_dfs(df, cfg)
                                for symbol, df in dfs_by_symbol.items()}

    pooled_stats: dict[str, dict] = {}
    for indicator_key, event_col, direction, label in SIGNAL_SPECS:
        per_symbol = {}
        for symbol, dfs in indicator_dfs_by_symbol.items():
            ind_df = dfs.get(indicator_key)
            if ind_df is None or event_col not in ind_df.columns:
                continue
            per_symbol[symbol] = event_returns(
                ind_df, event_col, horizon=cfg.target_horizon, direction=direction,
                entry_lag=cfg.entry_lag, entry_col=cfg.entry_col,
                cost_bps=cfg.cost_bps, slippage_bps=cfg.slippage_bps)

        vals, cluster_ids = pool_event_returns(per_symbol)
        if len(vals) == 0:
            pooled_stats[label] = dict(n=0, mean=float("nan"), t_stat=float("nan"))
            continue
        mean, t_stat = cluster_robust_t_stat(vals, cluster_ids)
        pooled_stats[label] = dict(n=len(vals), mean=mean, t_stat=t_stat)

    return pooled_stats


def scan_universe(dfs_by_symbol: dict[str, pd.DataFrame], cfg) -> tuple[pd.DataFrame, dict]:
    """
    반환: (scan_df, pooled_stats)
      scan_df   : 종목별 SymbolVerdict.to_row() 결과를 strength 내림차순으로
                  정렬한 DataFrame. run_advisor.py의 --mode query가 이 표에서
                  질문받은 종목을 뺀 나머지 중 verdict != "보유"인 것들을
                  "다른 유망 종목"으로 보여준다.
      pooled_stats : compute_pooled_stats()의 결과 그대로(캐시 저장용).
    """
    pooled_stats = compute_pooled_stats(dfs_by_symbol, cfg)

    rows = []
    for symbol, df in dfs_by_symbol.items():
        verdict = evaluate_symbol(df, symbol, cfg, pooled_stats=pooled_stats)
        rows.append(verdict.to_row())

    scan_df = pd.DataFrame(rows)
    if not scan_df.empty:
        scan_df = scan_df.sort_values("strength", ascending=False).reset_index(drop=True)
    return scan_df, pooled_stats
