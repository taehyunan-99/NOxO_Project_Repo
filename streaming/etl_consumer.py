from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from kafka import KafkaConsumer
from sqlalchemy import create_engine, text

from streaming.sensor_csv import load_bootstrap_rows, parse_measured_at


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_FILE = PROJECT_ROOT / "database" / "sensor_data_stream.sql"
TABLE_NAME = "sensor_data_stream"
BOOTSTRAP_RESET_EVENT = "bootstrap_reset"

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
TOPIC = os.getenv("KAFKA_SENSOR_TOPIC", "noxo.sensor.raw")
GROUP_ID = os.getenv("KAFKA_ETL_CONSUMER_GROUP_ID", "noxo-stream-etl")
BOOTSTRAP_FILE = os.getenv("KAFKA_BOOTSTRAP_FILE")
BOOTSTRAP_MINUTES = int(os.getenv("KAFKA_BOOTSTRAP_MINUTES", "15"))
BOOTSTRAP_ENABLED = os.getenv("STREAM_ETL_BOOTSTRAP_ENABLED", "true").lower() == "true"
POLL_TIMEOUT_MS = int(os.getenv("KAFKA_ETL_CONSUMER_TIMEOUT_MS", "0"))
RETRY_DELAY_SECONDS = int(os.getenv("STREAM_ETL_RETRY_DELAY_SECONDS", "5"))
AUTO_OFFSET_RESET = os.getenv("KAFKA_ETL_AUTO_OFFSET_RESET", "latest")

RAW_TO_DB_MAPPING = {
    "IGCC.CC.G1.ca_fqsg_cl": "syngas_flow",
    "IGCC.CC.G1.csgv": "igv_opening",
    "IGCC.CC.G1.NQKR3_MONITOR": "n2_offset",
    "IGCC.CC.G1.nicvs1": "n2_valve_1",
    "IGCC.CC.G1.FSAGR": "syngas_srv",
    "IGCC.CC.G1.FSAG11": "syngas_gcv_1",
    "IGCC.CC.G1.FSAG11A": "syngas_gcv_1a",
    "IGCC.CC.G1.FSAG12": "syngas_gcv_2",
    "IGCC.CC.G1.CSBHX": "ibh_valve",
    "IGCC.CC.G1.NQJ": "n2_flow",
    "IGCC.DeNOX.AT_H1_901_PV": "nox_ppm",
    "IGCC.CC.G1.TTXM": "exhaust_temp",
    "IGCC.CC.G1.DWATT": "power_mw",
    "IGCC.CC.G1.VNPR_P": "npr_primary",
    "IGCC.DeNOX.AIT_H1_902": "o2_pct",
    "IGCC.CC.G1.FPSG": "fpsg",
    "IGCC.CC.G1.FTSG": "ftsg",
    "IGCC.CC.G1.LHVSYNDW_SCF": "lhvsyndw_scf",
    "IGCC.CC.G1.FPSG2": "fpsg2",
    "IGCC.CC.G1.FPSG3": "fpsg3",
    "IGCC.CC.G1.AFDPM": "afdpm",
    "IGCC.CC.G1.CTIM": "ctim",
    "IGCC.CC.G1.afpap": "afpap",
    "IGCC.CC.G1.CPD": "cpd",
    "IGCC.CC.G1.CTD": "ctd",
    "IGCC.CC.G1.tnh_v": "tnh_v",
    "IGCC.CC.G1.ATID": "atid",
    "IGCC.CC.G1.EXHMASS": "exhmass",
    "IGCC.CC.G1.FPGN1_SEL": "fpgn1_sel",
    "IGCC.CC.G1.FPGN2_SEL": "fpgn2_sel",
    "IGCC.CC.G1.ROUTPUT_32": "routput_32",
    "IGCC.CC.G1.VNPR_S": "vnpr_s",
    "IGCC.CC.G1.NPNJ": "npnj",
    "IGCC.CC.G1.NTNJ": "ntnj",
    "IGCC.CC.G1.NQJO2": "nqjo2",
    "IGCC.CC.G1.ndt1": "ndt1",
    "IGCC.CC.G1.NPNJ2": "npnj2",
    "IGCC.CC.G1.ROUTPUT_6": "routput_6",
    "IGCC.IG.PIC7069A.PV": "pic7069a_pv",
    "IGCC.IG.ZT7069B.PV": "zt7069b_pv",
    "IGCC.DeNOX.TT_H1_90123": "tt_h1_90123",
    "IGCC.CC.G1.itdp": "itdp",
    "IGCC.CC.G1.tcsph1": "tcsph1",
}

DB_COLUMNS = list(RAW_TO_DB_MAPPING.values())
DISTURBANCE_DB_COLUMNS = (
    "fpsg",
    "ftsg",
    "lhvsyndw_scf",
    "fpsg2",
    "fpsg3",
    "afdpm",
    "ctim",
    "afpap",
    "cpd",
    "ctd",
    "tnh_v",
    "atid",
    "exhmass",
    "fpgn1_sel",
    "fpgn2_sel",
    "routput_32",
    "vnpr_s",
    "npnj",
    "ntnj",
    "nqjo2",
    "ndt1",
    "npnj2",
    "routput_6",
    "pic7069a_pv",
    "zt7069b_pv",
    "tt_h1_90123",
    "itdp",
    "tcsph1",
)
OPTIONAL_DB_COLUMNS = {"o2_pct", *DISTURBANCE_DB_COLUMNS}
LINEAGE_COLUMNS = [
    "source_file",
    "stream_topic",
    "kafka_partition",
    "kafka_offset",
    "ingest_mode",
]


def _build_database_url_from_components() -> str | None:
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    database = os.getenv("POSTGRES_DB", "igcc_db")
    host = os.getenv("STREAM_ETL_POSTGRES_HOST") or os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("STREAM_ETL_POSTGRES_PORT") or os.getenv("POSTGRES_PORT", "5432")

    if not user or password is None:
        return None

    return (
        "postgresql+psycopg://"
        f"{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{port}/{quote(database, safe='')}"
    )


def _database_url_from_env() -> str | None:
    return (
        os.getenv("STREAM_ETL_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or _build_database_url_from_components()
    )


def get_database_url() -> str:
    database_url = _database_url_from_env()
    if database_url:
        return database_url

    try:
        load_dotenv()
    except PermissionError:
        pass

    database_url = _database_url_from_env()
    if not database_url:
        raise ValueError(
            "STREAM_ETL_DATABASE_URL or DATABASE_URL is required for stream ETL consumer"
        )
    return database_url


def build_engine():
    return create_engine(get_database_url())


def read_sql_statements(sql_file: Path) -> list[str]:
    cleaned_lines: list[str] = []
    for line in sql_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        cleaned_lines.append(line)

    sql_text = "\n".join(cleaned_lines)
    return [statement.strip() for statement in sql_text.split(";") if statement.strip()]


def ensure_stream_table(engine) -> None:
    statements = read_sql_statements(SQL_FILE)
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))


def transform_message_to_row(
    message: dict,
    *,
    stream_topic: str,
    ingest_mode: str,
    kafka_partition: int | None = None,
    kafka_offset: int | None = None,
) -> dict:
    values = message.get("values") or {}
    row = {
        "measured_at": parse_measured_at(message["measured_at"]),
        "source_file": message.get("source", "unknown"),
        "stream_topic": stream_topic,
        "kafka_partition": kafka_partition,
        "kafka_offset": kafka_offset,
        "ingest_mode": ingest_mode,
    }

    missing_tags: list[str] = []
    for raw_tag, db_column in RAW_TO_DB_MAPPING.items():
        value = values.get(raw_tag)
        if value is None:
            if db_column in OPTIONAL_DB_COLUMNS:
                row[db_column] = None
                continue
            missing_tags.append(raw_tag)
            continue
        row[db_column] = float(value)

    if missing_tags:
        raise ValueError(f"missing required tags: {missing_tags}")

    return row


def is_bootstrap_reset_message(message: dict) -> bool:
    return message.get("event_type") == BOOTSTRAP_RESET_EVENT


def upsert_stream_row(engine, row: dict) -> None:
    insert_columns = ["measured_at", *DB_COLUMNS, *LINEAGE_COLUMNS]
    column_sql = ",\n            ".join(insert_columns)
    value_sql = ",\n            ".join(f":{column}" for column in insert_columns)
    update_columns = [*DB_COLUMNS, *LINEAGE_COLUMNS]
    update_sql = ",\n            ".join(
        f"{column} = EXCLUDED.{column}" for column in update_columns
    )
    sql = text(
        f"""
        INSERT INTO {TABLE_NAME} (
            {column_sql}
        ) VALUES (
            {value_sql}
        )
        ON CONFLICT (measured_at) DO UPDATE
        SET
            {update_sql},
            ingested_at = CURRENT_TIMESTAMP
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, row)


def bootstrap_stream_table(engine) -> int:
    if not BOOTSTRAP_ENABLED:
        return 0

    inserted = 0
    for message in load_bootstrap_rows(BOOTSTRAP_FILE, minutes=BOOTSTRAP_MINUTES):
        row = transform_message_to_row(
            message,
            stream_topic=TOPIC,
            ingest_mode="bootstrap",
        )
        upsert_stream_row(engine, row)
        inserted += 1
    return inserted


def build_consumer() -> KafkaConsumer:
    options = {
        "bootstrap_servers": BOOTSTRAP_SERVERS,
        "group_id": GROUP_ID,
        "auto_offset_reset": AUTO_OFFSET_RESET,
        "enable_auto_commit": True,
        "value_deserializer": lambda raw: json.loads(raw.decode("utf-8")),
        "key_deserializer": lambda raw: raw.decode("utf-8") if raw else None,
    }
    if POLL_TIMEOUT_MS > 0:
        options["consumer_timeout_ms"] = POLL_TIMEOUT_MS

    return KafkaConsumer(TOPIC, **options)


def run_consumer() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    engine = build_engine()
    ensure_stream_table(engine)

    bootstrap_count = bootstrap_stream_table(engine)
    logger.info(
        "stream_etl_bootstrap_done table=%s count=%s minutes=%s",
        TABLE_NAME,
        bootstrap_count,
        BOOTSTRAP_MINUTES,
    )

    while True:
        consumer = None
        try:
            consumer = build_consumer()
            logger.info(
                "stream_etl_consumer_started topic=%s bootstrap=%s group=%s auto_offset_reset=%s",
                TOPIC,
                BOOTSTRAP_SERVERS,
                GROUP_ID,
                AUTO_OFFSET_RESET,
            )
            for record in consumer:
                if is_bootstrap_reset_message(record.value):
                    bootstrap_count = bootstrap_stream_table(engine)
                    logger.info(
                        "stream_etl_bootstrap_reset table=%s count=%s minutes=%s topic=%s partition=%s offset=%s",
                        TABLE_NAME,
                        bootstrap_count,
                        BOOTSTRAP_MINUTES,
                        record.topic,
                        record.partition,
                        record.offset,
                    )
                    continue
                row = transform_message_to_row(
                    record.value,
                    stream_topic=record.topic,
                    kafka_partition=record.partition,
                    kafka_offset=record.offset,
                    ingest_mode="stream",
                )
                upsert_stream_row(engine, row)
        except KeyboardInterrupt:
            logger.info("stream_etl_consumer_stopped_by_user")
            break
        except Exception as exc:
            logger.warning("stream_etl_consumer_failed err=%s", exc)
            time.sleep(RETRY_DELAY_SECONDS)
        finally:
            if consumer is not None:
                consumer.close()


if __name__ == "__main__":
    run_consumer()
