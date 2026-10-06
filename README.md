# Trading Signal Lab

앞서 다룬 Pine Script 지표 5개(기관 변동 캔들, Delta Trading, Big Sales, Ultimate RSI, VP Box)를
파이썬으로 재구현하고, 각 신호가 통계적으로 유의미한 예측력을 갖는지
**이벤트 스터디(event study)** 방식으로 검증하는 도구입니다.

포지션 관리·수수료·슬리피지까지 반영한 완전한 백테스터가 아니라, "이 신호가 뜬 뒤 N봉
동안 평균적으로 방향이 맞았는가"를 t-검정 수준으로 빠르게 스크리닝하는 데 목적이 있습니다.
5개 지표 중 실전에 쓸 만한 게 있는지 1차로 거르는 용도로 보시면 됩니다.

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
# 업비트 코인 (일봉 1000개)
python run_backtest.py --market upbit --symbol KRW-BTC --unit day --count 1000

# 업비트 코인 (60분봉)
python run_backtest.py --market upbit --symbol KRW-BTC --unit minute --minute-unit 60 --count 1500

# 국내 주식 (SK하이닉스, 2022~2026 일봉)
python run_backtest.py --market krx --symbol 000660 --start 20220101 --end 20260901

# 삼성전자
python run_backtest.py --market krx --symbol 005930 --start 20220101 --end 20260901

# 기관 변동 캔들의 거래량 이상치 판정을 "평균의 2배"(원본) 대신
# Kyle(1985) 기반 z-스코어(종목 고유 변동성 대비 95% 유의수준)로 바꿔서 비교
python run_backtest.py --market upbit --symbol KRW-BTC --unit day --count 1000 \
  --id-vol-mode zscore --id-z-threshold 1.96

# Jegadeesh & Titman(1993) 모멘텀 필터: 롱 이벤트는 상승모멘텀 상위 30%,
# 숏 이벤트는 하락모멘텀 상위 30% 구간에서만 인정하도록 5개 지표 전부 필터링
python run_backtest.py --market upbit --symbol KRW-BTC --unit day --count 1000 \
  --mom-filter --mom-threshold 0.7
```

## 결과물 (output/ 폴더)

- `summary.csv` — 지표 × 방향(LONG/SHORT) × 보유기간(봉수)별 평균수익률·승률·t-stat·p-value·
  `significant_bh` 표. 엑셀로 열어서 피벗해보시면 편합니다.
- `event_curves.png` — 신호 발생 후 시간이 지날수록 평균 누적수익률이 어떻게 움직이는지
  지표별로 겹쳐 그린 그래프. 곡선이 꾸준히 우상향(롱)/우하향(숏)하면 신호에 실제로
  예측력이 있다는 뜻이고, 0 근처에서 지그재그하면 노이즈일 가능성이 높습니다.

터미널에도 실행 즉시 "보유기간 N봉 기준 t-stat 랭킹" 표가 출력됩니다. t-stat 절댓값이
대략 2 이상이면 통계적으로 눈여겨볼 만하지만, `n_events`(표본 수)가 한 자릿수면 우연일
가능성이 커서 별도로 걸러서 보셔야 합니다. 여러 지표·방향·보유기간을 한꺼번에 비교할
때는 개별 t-stat만 보지 말고 `significant_bh` 컬럼(아래 다중 검정 보정 절 참고)을
우선 확인하세요.

## 각 지표가 실제로 검증하는 신호

| 지표 | LONG 이벤트 | SHORT 이벤트 |
|---|---|---|
| 기관 변동 캔들 (Institutional Displacement) | 거래량 급등 + 몸통 큰 양봉 | 거래량 급등 + 몸통 큰 음봉 |
| Delta Trading | EMA(12/26) 골든크로스 + 거래량 확인 | EMA 데드크로스 + 거래량 확인 |
| Big Sales | 최근 N봉 내내 상승 마감이 없다가 거래량 폭증 | 최근 N봉 내내 하락 마감이 없다가 거래량 폭증 |
| Ultimate RSI | 과매도(20) 구간에서 위로 이탈(반등) | 과매수(80) 구간에서 아래로 이탈(반락) |
| VP Box | 직전 기간 POC(최다거래가) 가격까지 내려왔다 지지 확인 | 직전 기간 POC 가격까지 올라왔다 저항 확인 |

## 원본 Pine Script와의 차이 (근사한 부분)

- **기관 변동 캔들**: 원본은 하위 타임프레임(1분봉 등) 실체결로 매수/매도를 나누지만,
  여기서는 이벤트 판정(거래량 급등 + 몸통 비율)까지는 원본과 100% 동일하게 구현했고,
  캔들 내부 매수/매도 분할 표시만 생략했습니다(백테스트 신호 자체엔 영향 없음).
  거래량 이상치 판정은 `--id-vol-mode`로 원본 방식(`multiple`, 평균×배수)과
  Kyle(1985) 정보거래 모형에서 유도한 `zscore`(종목별 거래량 변동성 대비 z-스코어,
  `--id-z-threshold`로 임계값 조절) 중 선택할 수 있습니다. 이론적 근거는
  `delta-institutional-combo-분석.md` 참고.
- **Delta Trading**: EMA 크로스 + 거래량 확인 로직은 원본과 동일. 캔들 강조·테이블 시각화는 제외.
- **VP Box**: 원본은 실시간으로 박스를 그리지만, 여기서는 "완료된 이전 기간의 POC를 다음
  기간에 지지/저항으로 사용"하는 방식으로 단순화했습니다. bins/기간(`--vp-bins`, `--vp-freq`)은
  원본과 동일하게 조절 가능합니다.

## 모멘텀 필터 (`--mom-filter`)

Jegadeesh & Titman(1993)의 모멘텀(추세지속) 논리를 기존 5개 지표의 이벤트에 대한
사전 필터로 적용합니다. 원 논문은 매달 전체 종목을 과거 J개월 수익률로 **횡단면
(cross-sectional)** 정렬해서 승자/패자 10분위를 만들지만, 여기서는 종목 하나씩
시계열로 백테스트하므로 그 정렬이 불가능합니다 — 대신 "그 종목 자신의 과거
window 기간 모멘텀 분포 대비 지금이 몇 퍼센타일인가"라는 시계열 근사
(`indicators/momentum.py`)를 씁니다.

- 롱(bull) 이벤트는 `mom_percentile >= threshold`(상승모멘텀 상위권)일 때만 인정
- 숏(bear) 이벤트는 `mom_percentile <= 1-threshold`(하락모멘텀 상위권)일 때만 인정
- `--mom-lookback`(모멘텀 측정 구간, 기본 126봉≈6개월), `--mom-window`(퍼센타일
  롤링 윈도우, 기본 252봉≈1년), `--mom-threshold`(기본 0.7 = 상하위 30%)로 조절

필터 전/후 이벤트 개수가 터미널에 출력되니, 필터를 걸었을 때 표본이 너무 줄어들어
(`n_events`가 한 자릿수) t-stat이 오히려 불안정해지지는 않는지 같이 확인하세요.

## 한국시장 SMB·HML 팩터 (`factors/kr_fama_french.py`)

Fama & French(1993) 3팩터 모형의 사이즈(SMB)·가치(HML) 팩터를 한국시장(KRX) 데이터로
직접 구성하는 코드입니다. 지금 지표들의 이벤트 스터디 초과수익이 순수한 신호
예측력인지, 아니면 소형주·가치주 팩터 프리미엄이 재포장된 것인지 가르는 데 씁니다.

- KOSPI≈NYSE, KOSDAQ≈NASDAQ 대응관계로 사이즈·BE/ME breakpoint를 KOSPI 종목만으로
  계산(원 논문이 NYSE 종목만으로 breakpoint를 잡는 이유와 동일한 논리)
- BE/ME는 회계데이터 대신 pykrx가 주는 PBR의 역수(1/PBR)로 근사
- 원 논문은 연 1회(6월 말) 리밸런싱하지만, 여기서는 월말 리밸런싱으로 단순화
- 근사/단순화한 부분은 전부 `kr_fama_french.py` 상단 docstring에 명시

```bash
# SMB/HML 월별 시계열을 csv로 저장 (pykrx 네트워크 필요 — 로컬/서버에서 실행)
python build_kr_factors.py --start 20220101 --end 20240101 --output output/kr_factors.csv
```

`_smoke_test_factors.py`는 네트워크 없이, 가짜 데이터로 "SMB가 가치효과를 상쇄하고
순수 사이즈효과만 남기는지, HML이 사이즈효과를 상쇄하고 순수 가치효과만 남기는지"를
계산식 차원에서 검증합니다(`python _smoke_test_factors.py`로 실행).

같은 파일에 Carhart(1997) WML(모멘텀) 팩터도 구성돼 있습니다 — `compute_wml()`이
사이즈×형성기간모멘텀(t-12개월~t-1개월 수익률, 최근 1개월은 단기반전 회피를 위해
제외) 2×3 정렬로 승자-패자 롱숏 팩터를 만듭니다. `_smoke_test_wml.py`로 검증.

### 생존 편향(survivorship bias) 부분 완화 (`--delisted-return`)

리밸런싱 구간 [t0, t1] 도중 상장폐지·장기 거래정지된 종목은 t1 시점 조회에서
아예 사라집니다. 이전 동작은 이런 종목을 조용히 빼고 남은 종목끼리만 시가총액
가중치를 재정규화했는데, 상장폐지는 거의 항상 큰 손실로 끝나기 때문에 이렇게
빼면 SMB/HML/WML 수익률이 실제보다 좋게(위로) 편향됩니다.

```bash
# 구간 중 사라진 종목을 전손(-100%)으로 취급해 편향을 줄임
python build_kr_factors.py --start 20220101 --end 20240101 \
  --output output/kr_factors.csv --delisted-return -1.0
```

내부적으로 `_value_weighted_return(weight, close_t0, close_t1, delisted_return=None)`이
t0에는 있었지만 t1 조회에서 없는 종목을 어떻게 다룰지 정합니다. 기본값 `None`은
이전 동작(조용히 제외)과 동일하고, 숫자(예: `-1.0`)를 주면 t0 시점 가중치 비중은
그대로 유지한 채 그 종목에 지정한 수익률을 강제로 대입합니다. `compute_smb_hml()`/
`compute_wml()` 둘 다 같은 이름의 `delisted_return` 파라미터로 전달받습니다.

주의할 한계: 이건 "구간 중간에 사라진 종목을 조용히 빼지 않는다"는 부분 완화이지,
생존 편향을 완전히 없애는 건 아닙니다. `fetch_cross_section()`이 애초에 그 시점에
pykrx로 조회 자체가 안 되는 종목까지 포착하지는 못합니다. 수치 정확성과 편향
방향(전손 반영 시 SMB·HML이 실제로 낮아지는지)은 `_smoke_test_survivorship.py`로
검증했습니다.

## 팩터 조정 알파 (`backtest/engine.py::factor_adjust`, `event_alpha`)

이벤트 스터디로 얻은 신호별 수익률을 SMB/HML/WML에 OLS 회귀시켜, 팩터로 설명되지
않는 순수 알파(및 그 t-통계량)를 뽑아내는 함수입니다. 파이프라인의 마지막 검증
단계로, "5개 지표의 예측력이 진짜인지, 알려진 팩터 프리미엄의 재포장인지"를
가릅니다.

- `factor_adjust(event_dates, event_returns, factor_df)` — 저수준 회귀 함수. 이벤트
  날짜를 `factor_df`(월별 SMB/HML/WML)에서 가장 가까운 리밸런싱 시점과
  매칭(`merge_asof`)한 뒤 회귀. scipy/statsmodels 없이 numpy만으로 계수·t-통계량
  직접 계산.
- `event_alpha(df, event_col, factor_df, horizon=20)` — `event_study()`와 같은
  인터페이스(`df` + `event_col`)로 바로 쓸 수 있게 감싼 편의 함수.
- `factors/kr_fama_french.py`의 `combine_factors(smb_hml_df, wml_df)`로 SMB/HML과
  WML을 하나의 `factor_df`로 합칠 수 있습니다.

`_smoke_test_factor_adjust.py`는 노이즈 없이 정확한 선형관계로 만든 합성 데이터를
넣어서, 회귀가 넣은 그대로의 alpha·베타를 오차 1e-9 이내로 복원하고 R²≈1이
나오는지 검증합니다. 아직 `run_backtest.py` CLI에는 연결하지 않았고, 함수만
독립적으로 준비된 상태입니다.

## 거래비용·체결 시점 반영 (`--entry-lag`, `--entry-col`, `--cost-bps`)

이벤트 스터디가 "신호가 뜨자마자 그 봉 종가로 즉시, 비용 없이 체결된다"고
가정하면 두 가지 문제가 생깁니다.

- **Look-ahead bias**: 신호(예: 골든크로스)는 그 봉이 마감돼야 확정되는데, 같은
  봉 종가로 진입한다고 가정하면 사실상 그 봉이 끝나기 전의 정보를 이미 알고
  거래한 것과 같습니다. 현실적으로는 "t봉 종가에 신호 확정 → t+1봉에서 진입"이
  맞습니다.
- **거래비용 미반영**: 업비트(편도 0.05%×2=10bp), KRX(편도 수수료+거래세 약
  0.18%×2=36bp) 같은 왕복 비용을 빼지 않으면, 특히 짧은 보유기간(1~5봉) 신호일수록
  통계적 유의성이 실제보다 부풀려집니다.

```bash
# t봉 종가에 신호 확정 -> t+1봉 시가로 진입, KRX 왕복비용(36bp) 반영
python run_backtest.py --market krx --symbol 005930 --start 20220101 --end 20260901 \
  --entry-lag 1 --entry-col open --cost-bps 36

# 업비트는 왕복 10bp 근사
python run_backtest.py --market upbit --symbol KRW-BTC --unit day --count 1000 \
  --entry-lag 1 --entry-col open --cost-bps 10
```

내부적으로는 `backtest/engine.py::forward_return(df, horizon, entry_lag=0, entry_col=None, cost_bps=0.0)`이
`exit_price/entry_price - 1`을 계산할 때 `entry_lag>0`이면 진입가를 그만큼
미래로 밀어서(`entry_col` 지정 시 그 컬럼, 기본은 종가) 가져오고, `cost_bps`는
`direction`(롱/숏)과 무관하게 항상 차감합니다(비용은 포지션 방향과 상관없이 실제로
나가는 돈이기 때문 — 숏이라고 비용이 이익으로 바뀌면 안 됩니다). 이 값들은
`event_study()`·`compare_all()`·`event_alpha()`·`judge_indicator()`까지 전부
동일하게 관통해서 전달되므로, [3]단계 원 신호 검증과 [5]단계 알파 검증이 항상
같은 체결 가정을 씁니다. **기본값(entry_lag=0, entry_col=None, cost_bps=0.0)은
이전 버전의 동작과 완전히 동일**하므로 기존 스모크 테스트는 전부 그대로
통과하고, 새 옵션은 명시적으로 켜야만 적용됩니다. 수치 정확성은
`_smoke_test_cost_entry.py`(결정적 등비수열 가격으로 손 계산 값과 대조)로
검증했습니다.

## 다중 검정 보정 (`--fdr`, `backtest/engine.py::benjamini_hochberg`)

5개 지표 × 롱/숏 × 여러 보유기간(예: 1,3,5,10,20봉)을 한 번에 검증하면, 신호별
t-stat 하나하나는 정확해도 그중 몇 개는 순전히 우연으로 |t|>=2가 나올 수 있습니다
(가설을 30번 테스트하면 유의수준 5%에서 우연히 하나쯤은 "유의하다"고 나올 확률이
꽤 큽니다 — p-hacking의 전형적인 메커니즘). Benjamini-Hochberg(BH) 절차는 여러 개의
p-value를 동시에 볼 때 이런 우연한 발견의 비율(false discovery rate, FDR)을
목표 수준 이하로 통제하도록 유의성 판정 기준 자체를 보정해줍니다.

```bash
# 기본 FDR 10%. compare_all() 한 번의 호출(= 이번 실행에서 검증한 전체
# 신호x방향x보유기간 조합)을 하나의 검정 묶음으로 보고 보정한다.
python run_backtest.py --market krx --symbol 005930 --start 20220101 --end 20260901 --fdr 0.10
```

`compare_all()`이 만드는 요약표에 두 컬럼이 추가됩니다.
- `p_value` — `t_stat`으로부터 표준정규분포 근사(scipy 없이 `math.erfc`로 직접
  계산)로 구한 양측검정 p-value. t-분포가 아니라 정규분포로 근사하므로, 표본이
  아주 작으면(예: n_events가 한 자릿수) 실제보다 p-value를 약간 낙관적으로 볼 수
  있다는 점에 유의.
- `significant_bh` — 요약표 전체의 `p_value`에 BH 절차를 적용해 목표 FDR
  이하로 다중 검정을 보정한 뒤에도 유의하다고 남는 신호만 `True`. 개별 `t_stat`이
  |2| 이상이어도 `significant_bh`가 `False`면, 여러 조합을 동시에 비교한 상황을
  감안했을 때는 우연일 가능성을 배제할 수 없다는 뜻입니다.

`_smoke_test_bh.py`가 세 가지를 검증합니다: `p_value_from_t()`가 t=1.96 근방에서
p≈0.05를 정확히 내는지, `benjamini_hochberg()`가 교과서적 BH 절차(순위별 임계값
i/m×FDR, 조건을 만족하는 가장 큰 순위까지 전부 채택)를 정확히 재현하는지, 그리고
`compare_all()`의 요약표에 두 컬럼이 실제로 붙고 `significant_bh=True`인 행은
반드시 `p_value<=fdr`을 만족하는지.

주의: 이 보정은 `compare_all()`(파이프라인 3단계) 요약표 차원에서 이뤄집니다.
`decision/rule_based.py`의 `judge_indicator()`는 지표 하나씩 개별 호출되는
구조라 BH 절차를 적용할 "전체 묶음"이 그 안에 없어서, 아직 다중 검정 보정을
반영하지 않습니다 — 결과를 최종 채택하기 전에는 `run_backtest.py`가 출력하는
`significant_bh` 컬럼도 함께 확인하는 걸 권합니다.

## 파일 구조

```
data/           시세 수집 (업비트 REST API, pykrx/yfinance)
indicators/     5개 지표 파이썬 재구현 (각 파일 상단에 원본 대비 근사 사항 명시)
backtest/       이벤트 스터디 엔진 (forward_return, t-검정, 요약표 생성)
run_backtest.py CLI 진입점
_smoke_test.py  네트워크 없이 합성 데이터로 파이프라인이 정상 동작하는지 확인하는 자체 테스트
                (python _smoke_test.py 로 실행 — 실제 데이터 없이도 코드가 안 깨지는지만 확인)
```

## 다음에 고려해볼 만한 것

- 여기서 t-stat이 괜찮게 나온 신호가 있으면, 그걸 지난번 만든 Pine Script
  `Delta_Institutional_Combo_Alert.pine`처럼 실제 트레이딩뷰 알림용 코드로 다시 옮겨서
  TVExtBot 웹훅에 연결하는 흐름으로 이어갈 수 있습니다.
- 지금은 단일 신호 이벤트 스터디이므로, 여러 신호를 AND/OR로 조합했을 때 t-stat이
  더 좋아지는지도 다음 단계로 검증해볼 만합니다.
- 확률미분방정식 공부와 연결짓고 싶으시면, `forward_return`의 분포를 정규분포 가정 없이
  부트스트랩으로 재검정하거나, GBM 대비 실제 수익률 분포의 두꺼운 꼬리(fat tail)를
  비교하는 방향으로 확장할 수도 있습니다.
