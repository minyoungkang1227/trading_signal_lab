"""
텔레그램 봇으로 알림 메시지를 보내는 최소 기능. python-telegram-bot 같은 별도
패키지 없이 requests(이미 requirements.txt에 있음)만으로 Bot API를 직접 호출한다
— 알림 전송 하나만을 위해 새 의존성을 늘리지 않기 위함.

사용 전 준비(사람이 한 번만 하면 되는 설정, 코드가 대신해줄 수 없는 부분):
  1) 텔레그램에서 @BotFather를 찾아 대화 시작 -> /newbot -> 이름 정하기
     -> 발급되는 "봇 토큰"(숫자:영숫자 형태) 복사 = TELEGRAM_BOT_TOKEN
  2) 방금 만든 내 봇을 텔레그램에서 검색해 아무 메시지나 한 번 보내기
     (봇이 먼저 나에게 말을 걸 수는 없고, 내가 먼저 말을 걸어야 채팅방이 생김)
  3) 브라우저로 https://api.telegram.org/bot<TOKEN>/getUpdates 접속
     -> 응답 json에서 "chat":{"id": ...} 의 숫자값 = TELEGRAM_CHAT_ID
  4) GitHub 저장소 Settings -> Secrets and variables -> Actions 에
     TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID 두 개를 등록
     (코드/깃허브 공개 저장소에는 절대 직접 적지 않는다 — 노출되면 즉시 재발급 필요한
     보안 사고로 취급할 것)
"""
from __future__ import annotations

import os

import requests

_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


def send_telegram(text: str, token: str | None = None, chat_id: str | None = None,
                   timeout: float = 10.0) -> None:
    """토큰/chat_id를 안 주면 환경변수 TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID를 쓴다.
    (GitHub Actions에서는 워크플로우가 Secrets를 이 환경변수로 넣어준다.)

    실패(네트워크 오류, 잘못된 토큰 등)해도 예외를 삼키지 않고 그대로 올린다 —
    알림이 조용히 안 보내진 걸 모르고 넘어가는 게 가장 나쁜 실패 모드이므로,
    호출부(check_signals.py)가 이 예외를 보고 로그에 남기게 한다.
    """
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID가 설정되지 않았습니다 "
            "(환경변수 또는 함수 인자로 전달 필요)."
        )

    resp = requests.post(
        _API_BASE.format(token=token),
        json={"chat_id": chat_id, "text": text},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("ok"):
        raise RuntimeError(f"텔레그램 전송 실패: {payload}")
