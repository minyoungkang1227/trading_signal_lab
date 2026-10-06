"""
신호 알림(텔레그램) 기능. trading_signal_lab 본체(run_backtest.py, backtest/engine.py,
indicators/*)는 전혀 건드리지 않고, 그 함수들을 그대로 재사용하는 별도 레이어다
(app.py와 동일한 '얇은 껍데기' 원칙).

구성:
  config.py        watchlist.json 로딩 + run_backtest.py argparse 기본값과 1:1로
                    맞춘 DEFAULTS/build_args (app.py의 것과 동일한 패턴, 독립 복제).
  telegram.py       텔레그램 봇 메시지 전송 (requests 기반, 외부 의존성 추가 없음).
  state.py          "오늘 이미 보낸 알림"을 기억해서 같은 신호를 하루에 여러 번
                    보내지 않게 하는 상태 파일 입출력.
  check_signals.py  위 세 개를 엮어 "오늘 watchlist 중 새로 뜬 신호가 있으면
                    텔레그램으로 보낸다"를 1회 실행하는 진입점. GitHub Actions
                    스케줄(.github/workflows/signal_alert.yml)이 매일 이 스크립트를
                    돌린다.
"""
