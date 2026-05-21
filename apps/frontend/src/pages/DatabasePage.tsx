/**
 * DB 스키마 페이지.
 *
 * 본 페이지는 batch ETL, streaming ETL, simulation log 저장 흐름에서
 * 프론트 팀이 알아야 하는 DB 구조를 요약한다.
 */

/**
 * 식별자 표기 원칙:
 * - 운영 sensor_data는 train CSV(`NOx_train_*.csv`)만 적재한다.
 * - test CSV(`NOx_test_20250825.csv`)는 Kafka 스트리밍 시뮬레이션 입력으로 분리한다.
 * - `co`는 학습 타겟에서 제외되어 운영 테이블/로그에 추가하지 않는다.
 */
type ColumnSpec = readonly [name: string, type: string, constraint: string, description: string]

type ErdTable = {
  name: string
  note: string
  columns: readonly ColumnSpec[]
}

const erdTables: readonly ErdTable[] = [
  {
    name: 'sensor_data',
    note: 'Batch ETL — train/historical',
    columns: [
      ['measured_at', 'timestamp', 'PK, NOT NULL', '측정 시간'],
      ['syngas_flow', 'double precision', 'NOT NULL', '합성가스 유량'],
      ['igv_opening', 'double precision', 'NOT NULL', 'IGV 개도'],
      ['n2_offset', 'double precision', 'NOT NULL', 'N2 오프셋'],
      ['n2_valve_1', 'double precision', 'NOT NULL', 'N2 주입 제어밸브 1 개도'],
      ['syngas_srv', 'double precision', 'NOT NULL', 'Syngas SRV 개도'],
      ['syngas_gcv_1', 'double precision', 'NOT NULL', 'Syngas GCV 1 개도'],
      ['syngas_gcv_1a', 'double precision', 'NOT NULL', 'Syngas GCV 1A 개도'],
      ['syngas_gcv_2', 'double precision', 'NOT NULL', 'Syngas GCV 2 개도'],
      ['ibh_valve', 'double precision', 'NOT NULL', 'IBH 입구 가열 제어밸브 개도'],
      ['n2_flow', 'double precision', 'NOT NULL', 'N2 주입 유량'],
      ['nox_ppm', 'double precision', 'NOT NULL', 'NOx 농도'],
      ['exhaust_temp', 'double precision', 'NOT NULL', '배기가스 온도'],
      ['power_mw', 'double precision', 'NOT NULL', '발전기 출력'],
      ['npr_primary', 'double precision', 'NOT NULL', '1차 노즐 압력비'],
    ],
  },
  {
    name: 'sensor_data_stream',
    note: 'Stream ETL — test/bootstrap/live',
    columns: [
      ['id', 'bigserial', 'PK', '스트림 적재 행 ID'],
      ['measured_at', 'timestamp', 'NOT NULL, UNIQUE', '원천 센서 측정 시간'],
      ['운영 컬럼 14개 그룹', 'double precision', 'NOT NULL', 'sensor_data와 동일한 핵심 운전 컬럼 묶음'],
      ['o2_pct', 'double precision', 'nullable', 'NOx 15% O2 보정용 선택 컬럼'],
      ['ML 보조 피처 28개 그룹', 'double precision', 'nullable', '예측 보조용 disturbance/raw 피처 컬럼 묶음'],
      ['source_file', 'varchar(255)', 'NOT NULL', '입력 원천 파일명'],
      ['stream_topic', 'varchar(128)', 'NOT NULL', 'Kafka-compatible topic 이름'],
      ['kafka_partition', 'integer', 'UNIQUE 조합', 'Kafka partition'],
      ['kafka_offset', 'bigint', 'UNIQUE 조합', 'Kafka offset'],
      ['ingest_mode', 'varchar(16)', "CHECK ('bootstrap', 'stream')", '초기 적재/실시간 적재 구분'],
      ['ingested_at', 'timestamp', 'NOT NULL, default now()', 'DB 적재 시각'],
    ],
  },
  {
    name: 'simulation_session_log',
    note: 'Backend persistence',
    columns: [
      ['id', 'bigint', 'PK', '세션 로그 ID'],
      ['sid', 'varchar(64)', 'UNIQUE', '세션 식별자'],
      ['started_at', 'timestamp', 'NOT NULL', '세션 시작 시간'],
      ['ended_at', 'timestamp', 'nullable', '세션 종료 시간'],
      ['notes', 'text', 'nullable', '비고'],
    ],
  },
  {
    name: 'simulation_input_log',
    note: 'Backend persistence',
    columns: [
      ['id', 'bigint', 'PK', '입력 이력 ID'],
      ['sid', 'varchar(64)', 'FK', 'simulation_session_log.sid 참조'],
      ['created_at', 'timestamp', 'NOT NULL', '입력 기록 시간'],
      ['syngas_flow', 'double precision', 'NOT NULL', '합성가스 유량 목표값'],
      ['igv_opening', 'double precision', 'NOT NULL', 'IGV 개도 목표값'],
      ['n2_offset', 'double precision', 'NOT NULL', 'N2 오프셋 목표값'],
      ['n2_valve_1', 'double precision', 'NOT NULL', 'N2 주입 제어밸브 1 개도 목표값'],
      ['syngas_srv', 'double precision', 'NOT NULL', 'Syngas SRV 개도 목표값'],
      ['syngas_gcv_1', 'double precision', 'NOT NULL', 'Syngas GCV 1 개도 목표값'],
      ['syngas_gcv_1a', 'double precision', 'NOT NULL', 'Syngas GCV 1A 개도 목표값'],
      ['syngas_gcv_2', 'double precision', 'NOT NULL', 'Syngas GCV 2 개도 목표값'],
      ['ibh_valve', 'double precision', 'NOT NULL', 'IBH 입구 가열 제어밸브 개도 목표값'],
      ['n2_flow', 'double precision', 'NOT NULL', 'N2 주입 유량 목표값'],
    ],
  },
  {
    name: 'forecast_log',
    note: 'Backend persistence',
    columns: [
      ['id', 'bigint', 'PK', '예측 이력 ID'],
      ['sid', 'varchar(64)', 'FK', 'simulation_session_log.sid 참조'],
      ['created_at', 'timestamp', 'NOT NULL', '예측 생성 시간'],
      ['target_time', 'timestamp', 'NOT NULL', '예측 대상 미래 시점'],
      ['predicted_nox', 'double precision', 'NOT NULL', '예측된 NOx 농도'],
      ['threshold_value', 'double precision', 'NOT NULL', '예측 시점의 NOx 임계값 스냅샷'],
      ['threshold_exceeded', 'boolean', 'NOT NULL', '임계값 초과 여부'],
    ],
  },
]

export function DatabasePage() {
  return (
    <main className="content-page">
      <div className="content-inner db-inner">
        <div className="section-label">DATABASE</div>
        <h1 className="section-title">데이터 모델 구조</h1>
        <div className="db-summary">
          <p>
            IGCC train 데이터는 batch 테이블에, test-day replay 데이터는 streaming 전용 테이블에 분리 저장한다.
          </p>
          <p>
            시뮬레이션 세션·제어 입력·예측 결과는 sid 기준 로그 테이블로 연결한다.
          </p>
        </div>

        <section
          className="panel"
          style={{
            padding: '14px 18px',
            margin: '12px 0 24px',
            borderColor: 'rgba(245, 158, 11, 0.45)',
            background: 'rgba(245, 158, 11, 0.08)',
          }}
        >
          <strong>구현 상태</strong> — `sensor_data`는 Airflow batch ETL의 train 적재 테이블이고,
          `sensor_data_stream`은 Redpanda 기반 streaming simulation 데이터를 lineage와 함께 저장한다.
          `simulation_session_log`, `simulation_input_log`, `forecast_log`는 백엔드 세션/예측 로그 영속화에 사용한다.
        </section>

        <section className="erd-container">
          <div className="erd-wrap">
            <svg className="erd-svg" viewBox="0 0 1120 620">
              <line x1="700" y1="320" x2="790" y2="250" className="erd-link" />
              <line x1="700" y1="365" x2="790" y2="490" className="erd-link" />
              <text x="710" y="295" className="erd-link-label">sid 1:N</text>
              <text x="710" y="435" className="erd-link-label">sid 1:N</text>
            </svg>
            <TableNode
              left={24}
              top={24}
              width={300}
              title="sensor_data"
              rows={[
                '[PK] measured_at',
                'train/historical batch rows',
                'ControlVars 10 columns',
                'nox_ppm, exhaust_temp, power_mw',
                'npr_primary',
              ]}
            />
            <TableNode
              left={24}
              top={310}
              width={330}
              title="sensor_data_stream"
              rows={[
                '[PK] id',
                '[UQ] measured_at',
                'test/bootstrap/live rows',
                'ControlVars + OutputVars',
                'optional ML feature columns',
                'topic, partition, offset',
                'ingest_mode, ingested_at',
              ]}
            />
            <TableNode
              left={430}
              top={245}
              width={270}
              title="simulation_session_log"
              rows={['[PK] id', '[UQ] sid', 'started_at', 'ended_at', 'notes']}
            />
            <TableNode
              left={790}
              top={145}
              width={300}
              title="simulation_input_log"
              rows={['[PK] id', '[FK] sid', 'created_at', 'ControlVars 10 columns']}
            />
            <TableNode
              left={790}
              top={430}
              width={300}
              title="forecast_log"
              rows={['[PK] id', '[FK] sid', 'created_at', 'target_time', 'predicted_nox', 'threshold_value', 'threshold_exceeded']}
            />
          </div>
        </section>

        <section className="content-section">
          <h2 className="section-title">테이블 명세</h2>
          <div className="spec-grid">
            {erdTables.map((table) => (
              <article key={table.name} className="panel spec-card">
                <header className="spec-header">
                  <div className="spec-title">{table.name}</div>
                  <span className="mono spec-count">{table.note}</span>
                </header>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>컬럼</th>
                      <th>타입</th>
                      <th>제약</th>
                      <th>설명</th>
                    </tr>
                  </thead>
                  <tbody>
                    {table.columns.map(([name, type, constraint, description]) => (
                      <tr key={`${table.name}-${name}`}>
                        <td className="label-cell">{name}</td>
                        <td>{type}</td>
                        <td className="muted-cell">{constraint}</td>
                        <td className="description-cell">{description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </article>
            ))}
          </div>
        </section>
      </div>
      <PageFooter />
    </main>
  )
}

function TableNode({
  left,
  top,
  width,
  title,
  rows,
}: {
  left: number
  top: number
  width: number
  title: string
  rows: string[]
}) {
  return (
    <div className="table-node" style={{ left, top, width }}>
      <div className="table-node-header">{title}</div>
      {rows.map((row) => (
        <div key={row} className="table-node-row">
          <span className="mono">{row}</span>
        </div>
      ))}
    </div>
  )
}

function PageFooter() {
  return (
    <footer className="page-footer">
      <span>NOxO · 합성가스 발전 NOx 디지털 트윈 · 2026-05-21</span>
      <div className="footer-links">
        <span>PRD</span>
        <span>Architecture</span>
        <span>Repo</span>
      </div>
    </footer>
  )
}
