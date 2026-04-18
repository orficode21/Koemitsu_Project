from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import LOGS_DIR, UPLOADS_DIR


def ensure_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def new_session_id() -> str:
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return f'KOE-{stamp}-{uuid.uuid4().hex[:6]}'


def session_folder(session_id: str) -> Path:
    path = UPLOADS_DIR / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if path.exists():
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)


def save_calibration_log(session_id: str, payload: dict[str, Any]) -> None:
    append_csv(LOGS_DIR / 'calibration_sessions.csv', {'session_id': session_id, **payload})


def save_note_result_log(session_id: str, payload: dict[str, Any]) -> None:
    append_csv(LOGS_DIR / 'note_results.csv', {'session_id': session_id, **payload})


def save_summary_log(session_id: str, payload: dict[str, Any]) -> None:
    append_csv(LOGS_DIR / 'session_summary.csv', {'session_id': session_id, **payload})


def save_session_json(session_id: str, calibration: dict[str, Any], results: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    ensure_dirs()
    path = LOGS_DIR / f'{session_id}.json'
    payload = {
        'session_id': session_id,
        'calibration': calibration,
        'results': results,
        'summary': summary,
    }
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')
