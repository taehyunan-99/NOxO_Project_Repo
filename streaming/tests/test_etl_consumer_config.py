from __future__ import annotations

import importlib


def test_stream_etl_database_url_takes_precedence(monkeypatch):
    monkeypatch.setenv("STREAM_ETL_DATABASE_URL", "postgresql+psycopg://stream/db")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://legacy/db")

    import streaming.etl_consumer as etl_consumer

    importlib.reload(etl_consumer)

    assert etl_consumer.get_database_url() == "postgresql+psycopg://stream/db"


def test_database_url_can_be_built_from_postgres_components(monkeypatch):
    monkeypatch.delenv("STREAM_ETL_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_USER", "root")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("POSTGRES_DB", "igcc_db")
    monkeypatch.setenv("STREAM_ETL_POSTGRES_HOST", "postgres")
    monkeypatch.setenv("STREAM_ETL_POSTGRES_PORT", "5432")

    import streaming.etl_consumer as etl_consumer

    importlib.reload(etl_consumer)

    assert (
        etl_consumer.get_database_url()
        == "postgresql+psycopg://root:secret@postgres:5432/igcc_db"
    )


def test_bootstrap_reset_message_detection():
    import streaming.etl_consumer as etl_consumer

    assert etl_consumer.is_bootstrap_reset_message(
        {"event_type": "bootstrap_reset"}
    )
    assert not etl_consumer.is_bootstrap_reset_message(
        {"measured_at": "2025-08-25 00:15:00", "values": {}}
    )


def test_o2_pct_is_optional_when_transforming_message():
    import streaming.etl_consumer as etl_consumer

    values = {
        raw_tag: 1.0
        for raw_tag in etl_consumer.RAW_TO_DB_MAPPING
        if raw_tag != "IGCC.DeNOX.AIT_H1_902"
    }
    row = etl_consumer.transform_message_to_row(
        {"measured_at": "2025-08-25 00:15:00", "values": values},
        stream_topic="noxo.sensor.raw",
        ingest_mode="stream",
    )

    assert row["o2_pct"] is None


def test_disturbance_columns_are_optional_when_transforming_message():
    import streaming.etl_consumer as etl_consumer

    missing_raw_tag = "IGCC.CC.G1.LHVSYNDW_SCF"
    values = {
        raw_tag: 1.0
        for raw_tag in etl_consumer.RAW_TO_DB_MAPPING
        if raw_tag != missing_raw_tag
    }
    row = etl_consumer.transform_message_to_row(
        {"measured_at": "2025-08-25 00:15:00", "values": values},
        stream_topic="noxo.sensor.raw",
        ingest_mode="stream",
    )

    assert row["lhvsyndw_scf"] is None
