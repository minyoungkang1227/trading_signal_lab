"""
"어떤 신호를 어느 날짜까지 이미 알렸는지" 기억하는 상태 파일(JSON) 입출력.

매일 같은 신호가 계속 True로 남아 있을 수 있어서(예: 과매도 상태가 며칠씩 지속),
그냥 "지금 True인가"만 보면 같은 신호를 매일 반복해서 보내게 된다. 그래서
"그 신호가 마지막으로 발생한 날짜가 저번에 알린 날짜보다 새로운가"를 기준으로
삼는다 — 이 파일은 그 "저번에 알린 날짜"를 기록해두는 역할만 한다.

GitHub Actions에서는 이 파일을 워크플로우가 실행 끝에 저장소로 다시 커밋해야
다음 실행에서도 기억이 유지된다(.github/workflows/signal_alert.yml 참고).
"""
from __future__ import annotations

import json
from pathlib import Path


def load_state(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: str | Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
