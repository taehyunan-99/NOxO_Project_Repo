"""sensor_data_stream 폴링 전용 repository.

KafkaSensorStream과 별개 경로 — kafka-etl-consumer가 적재한 row를
backend가 (ingested_at, id) composite cursor 기반으로 단조 증가 순서로 pull.

DDL은 `database/sensor_data_stream.sql` — 14 운영 컬럼 + o2_pct +
ML 외란 컬럼 + 5 lineage + ingested_at.
schema SoT 변경은 협의 필요(`AGENTS.md` root 가드).

도메인 키 매핑 — `app/domain/tags.py::ALL_TAGS_TO_DOMAIN`이 SoT.
DB 컬럼 vs 도메인 식별자가 일부 다르다 (`nox_ppm`/`power_mw`/`npr_primary`).
KafkaSensorStream 경로(`normalize_raw_message`)와 키 셋을 맞추지 않으면 RealtimeEngine
forecaster가 NOx feature 0 stagnation으로 무한 차단된다(PR #2 `_warmup_reason`).

DB connection pool 주의 — `DbContext.session_factory`는 `sensor_repo`/
`simulation_log_repo`와 동일 engine을 공유한다(pool_size=5). 1초 폴링은
1 connection만 점유하므로 문제 없으나, 부하 증가 시 별도 engine 분리 검토.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

# (DB 컬럼명, SensorBuffer 도메인 키) — DDL 순서 보존.
# 대부분 동일명이지만 nox_ppm/power_mw/npr_primary 3개는 변환 필요.
# tags.py::ALL_TAGS_TO_DOMAIN과 정합(`nox_ppm`→`nox`, `power_mw`→`power`,
# `npr_primary`→`vnpr_p` 외란 도메인 키). o2_pct는 NOx 15% O2 보정 표시용.
_STREAM_COLUMN_MAP: tuple[tuple[str, str], ...] = (
    ("syngas_flow", "syngas_flow"),
    ("igv_opening", "igv_opening"),
    ("n2_offset", "n2_offset"),
    ("n2_valve_1", "n2_valve_1"),
    ("syngas_srv", "syngas_srv"),
    ("syngas_gcv_1", "syngas_gcv_1"),
    ("syngas_gcv_1a", "syngas_gcv_1a"),
    ("syngas_gcv_2", "syngas_gcv_2"),
    ("ibh_valve", "ibh_valve"),
    ("n2_flow", "n2_flow"),
    ("nox_ppm", "nox"),
    ("exhaust_temp", "exhaust_temp"),
    ("power_mw", "power"),
    ("npr_primary", "vnpr_p"),
    ("o2_pct", "o2_pct"),
    ("fpsg", "fpsg"),
    ("ftsg", "ftsg"),
    ("lhvsyndw_scf", "lhvsyndw_scf"),
    ("fpsg2", "fpsg2"),
    ("fpsg3", "fpsg3"),
    ("afdpm", "afdpm"),
    ("ctim", "ctim"),
    ("afpap", "afpap"),
    ("cpd", "cpd"),
    ("ctd", "ctd"),
    ("tnh_v", "tnh_v"),
    ("atid", "atid"),
    ("exhmass", "exhmass"),
    ("fpgn1_sel", "fpgn1_sel"),
    ("fpgn2_sel", "fpgn2_sel"),
    ("routput_32", "routput_32"),
    ("vnpr_s", "vnpr_s"),
    ("npnj", "npnj"),
    ("ntnj", "ntnj"),
    ("nqjo2", "nqjo2"),
    ("ndt1", "ndt1"),
    ("npnj2", "npnj2"),
    ("routput_6", "routput_6"),
    ("pic7069a_pv", "pic7069a_pv"),
    ("zt7069b_pv", "zt7069b_pv"),
    ("tt_h1_90123", "tt_h1_90123"),
    ("itdp", "itdp"),
    ("tcsph1", "tcsph1"),
)

# composite cursor — (ingested_at, id) 단조성 보장.
# PostgreSQL CURRENT_TIMESTAMP은 transaction start time이라 동일 ms 내 tie 가능.
# ms 해상도 tie + LIMIT 경계에서 row가 영구 skip되는 위험을 id로 회피.
StreamCursor = tuple[datetime, int]


class SensorStreamRepository:
    """sensor_data_stream row를 (ingested_at, id) ASC로 polling.

    startup bootstrap은 backend가 CSV에서 직접 채운다. 이후 producer loop가
    bootstrap reset marker를 보낼 때 stream ETL이 bootstrap rows를 다시 upsert하므로
    poller는 bootstrap/stream 둘 다 읽어 세션 버퍼가 새 replay cycle을 따라가게 한다.
    """

    def __init__(self, db_session_factory: sessionmaker[Session]):
        self.session_factory = db_session_factory

    async def fetch_since(
        self,
        cursor: StreamCursor | None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """(ingested_at, id) > cursor 인 stream row를 ASC 정렬로 반환.

        cursor가 None이면 호출자가 latest_cursor()로 초기화한 뒤 호출한다.
        """
        if cursor is None:
            raise ValueError("cursor required (use latest_cursor)")
        return await asyncio.to_thread(self._fetch_sync, cursor, limit)

    async def latest_cursor(self) -> StreamCursor | None:
        """초기 cursor 결정용. 빈 테이블이면 None."""
        return await asyncio.to_thread(self._latest_sync)

    def _fetch_sync(self, cursor: StreamCursor, limit: int) -> list[dict[str, Any]]:
        db_cols = [db for db, _ in _STREAM_COLUMN_MAP]
        col_list = ", ".join(db_cols)
        # (ingested_at, id) 튜플 비교 — 두 컬럼 모두 ASC 정렬해 cursor 전진.
        sql = text(
            f"""
            SELECT id, measured_at, ingested_at, {col_list}
            FROM sensor_data_stream
            WHERE (ingested_at, id) > (:cursor_ts, :cursor_id)
            ORDER BY ingested_at ASC, id ASC
            LIMIT :limit
            """
        )
        cursor_ts, cursor_id = cursor
        with self.session_factory() as session:
            result = session.execute(
                sql,
                {
                    "cursor_ts": cursor_ts,
                    "cursor_id": int(cursor_id),
                    "limit": int(limit),
                },
            )
            rows = result.mappings().all()
        return [self._to_domain_dict(r) for r in rows]

    def _latest_sync(self) -> StreamCursor | None:
        sql = text(
            """
            SELECT ingested_at, id
            FROM sensor_data_stream
            ORDER BY ingested_at DESC, id DESC
            LIMIT 1
            """
        )
        with self.session_factory() as session:
            row = session.execute(sql).mappings().first()
        if row is None:
            return None
        ts = row["ingested_at"]
        if not isinstance(ts, datetime):
            return None
        return (ts, int(row["id"]))

    @staticmethod
    def _to_domain_dict(row: Any) -> dict[str, Any]:
        # DB 컬럼 → 도메인 키 변환 후 SensorBuffer 호환 dict로 반환.
        # ingested_at·id는 cursor 전진용으로 보존(poller가 strip), measured_at은 도메인 시각.
        out: dict[str, Any] = {
            domain_key: row[db_col]
            for db_col, domain_key in _STREAM_COLUMN_MAP
        }
        out["measured_at"] = row["measured_at"]
        out["ingested_at"] = row["ingested_at"]
        out["id"] = row["id"]
        return out
