"""디지털 트윈 시뮬레이션 설정값 중앙 관리 모듈.

외부에서 주입하거나 튜닝이 필요한 값들을 한 곳에서 관리한다.
물리 상수(8.314, 273.15 등)는 chemistry.py에 그대로 둔다.

카테고리:
1. OperatingPoint  : 기준 운전점 및 초기 상태
2. TimeConstants   : 변수별 1차 lag 시간상수 [초]
3. ThresholdConfig : NOx / 안전 임계치
4. SimStepConfig   : 시뮬 step 크기 및 블렌딩 비율
5. FeatureConfig   : 파생 피처 계산식 상수 (공기비, CO, 효율)
6. ChemistryConfig : Zeldovich 반응 상수 (실험적 조정 대상)
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================
# 1. 기준 운전점 / 초기 상태
# [DB 협의 필요] 실측 평균값으로 교체 예정
# ============================================================
@dataclass(frozen=True)
class OperatingPoint:
    """기준(정격) 운전점. 파생 피처 계산 기준값으로 사용된다.

    제어 변수 10개의 기준값(`BACKEND_ARCHITECTURE.md §7` 예시 기준).
    신규 7개 변수의 단위/한계는 [추후 결정] — 임의 가안값.
    """

    # 학습 데이터(NOx_test_20250825.csv, DWATT>50MW 정상 운전 구간) 실측 mean으로 보정.
    # 옛 가안값은 단위/스케일이 안 맞아 compute_lambda 등 파생 피처가 폭주(λ≈32)했음.
    syngas_flow: float = 43.0      # IGCC.CC.G1.ca_fqsg_cl 실측 mean 43.14
    igv_opening: float = 63.0      # IGCC.CC.G1.csgv 실측 mean 63.14
    n2_offset: float = -10.0       # IGCC.CC.G1.NQKR3_MONITOR 실측 mean -9.98 (부호 주의)
    n2_valve_1: float = 28.0       # IGCC.CC.G1.nicvs1 실측 mean 27.56
    syngas_srv: float = 39.0       # IGCC.CC.G1.FSAGR 실측 mean 38.72
    syngas_gcv_1: float = 73.0     # IGCC.CC.G1.FSAG11 실측 mean 72.79
    syngas_gcv_1a: float = 44.0    # IGCC.CC.G1.FSAG11A 실측 mean 43.85
    syngas_gcv_2: float = 15.0     # IGCC.CC.G1.FSAG12 실측 mean 14.96
    ibh_valve: float = 0.4         # IGCC.CC.G1.CSBHX 실측 mean 0.37 (정상 운전 시 거의 닫힘)
    n2_flow: float = 29.0          # IGCC.CC.G1.NQJ 실측 mean 29.25
    exhaust_temp: float = 627.0    # IGCC.CC.G1.TTXM 실측 mean 627.54
    air_flow: float = 4500.0       # IGV 63%일 때 추정 공기유량 [kg/h, 가안]


# ============================================================
# 2. 초기 출력 상태값
# [DB 협의 필요] 실측 정상상태 평균으로 교체 예정
# ============================================================
@dataclass(frozen=True)
class InitialOutput:
    """세션 시작 시 t=0 출력 초기값.

    `co`는 학습 타겟에서 제외(`REFACTOR_FLAME_TEMP_TO_EXHAUST_TEMP.md`).
    """

    nox: float = 20.0          # [ppm, 가안]
    exhaust_temp: float = 580.0  # [°C]
    lambda_: float = 1.10      # [무차원]
    efficiency: float = 0.89   # [무차원]
    power: float = 248.6       # [MW, 가안]
    nox_integrated: float = 20.0  # Zeldovich ODE 적분 초기값 [ppm]


# ============================================================
# 3. 1차 lag 시간상수
# [조사 필요] 실제 운전 데이터 회귀로 갱신 예정
# ============================================================
@dataclass(frozen=True)
class TimeConstants:
    """변수별 1차 lag 시간상수. 단위: 초.

    기존 3개(fuel/n2/igv) + 신규 7개 제어 변수 — 신규 변수 τ는 가안 1.0초.
    [조사 필요] 실제 운전 데이터 회귀로 갱신 예정.
    """

    # 입력(제어 변수) lag — 기존 3개
    fuel: float = 1.0    # 합성가스 유량 응답
    n2: float = 1.0      # 희석질소 응답
    igv: float = 2.0     # IGV 개도 응답 (기계적 액추에이터)

    # 입력(제어 변수) lag — 신규 7개 [가안]
    n2_valve_1: float = 1.0
    syngas_srv: float = 1.0
    syngas_gcv_1: float = 1.0
    syngas_gcv_1a: float = 1.0
    syngas_gcv_2: float = 1.0
    ibh_valve: float = 1.0
    n2_flow: float = 1.0

    # 출력(연소 결과) lag — `co` 제거
    temp: float = 10.0   # 배기온도 — 열관성으로 가장 느림
    nox: float = 5.0     # NOx 응답
    power: float = 8.5   # 발전량 응답


# ============================================================
# 4. NOx / 운전 안전 임계치
# [조사 필요] 실제 규제값 및 운전 한계로 교체 예정
# ============================================================
@dataclass(frozen=True)
class ThresholdConfig:
    """임계치 및 경고 기준값.

    nox_warning_ppm은 시뮬 엔진 warning 발화 + 프론트 NOx 경고선 둘 다 기준이 된다.
    그 외 운영 임계(효율/배기온/λ 운영 한계)는 [가안] — 도메인 전문가 검토 후 교체.
    """

    # 시뮬 엔진 내부 안전 클램프 — 운영자 화면과 무관, 변경 금지
    nox_ceiling_ppm: float = 1000.0   # Zeldovich ODE 물리적 상한 [ppm]
    nox_floor_ppm: float = 0.0        # NOx 음수 방지 floor [ppm]
    nox_rate_max_ppm_per_s: float = 50.0  # Zeldovich 생성률 폭주 방지 [ppm/s]
    lambda_min: float = 0.5           # 공기비 최소 클램프 (연료과잉 방지)

    # 운영 임계 — 프론트 화면 표시용. 시뮬 동작에는 영향 없음.
    nox_warning_ppm: float = 29.5     # NOx 경고 임계치 [ppm] — CSV 실측 mean+1.3σ≈p99

    # 발전 효율 운영 임계 — GE 7F 단순 사이클 스펙 ~38.5%(LHV) 기준.
    # 종전 가안(0.3633/0.3630)은 0.03%p 간격으로 센서 노이즈에 채터링 발생 → 도메인 검토 후 상향.
    efficiency_caution: float = 0.370
    efficiency_danger: float = 0.360

    # 배기온도(TTXM) 상한 운영 임계 [°C] — GE 7FA 부분부하 시 IGV 제어로 ~650℃ 부근 유지.
    # 종전 가안(633/638)은 정상 운전 최댓값(639.76℃)보다 낮아 오알람 위험 → 도메인 검토 후 상향.
    exhaust_caution_c: float = 642.0
    exhaust_danger_c: float = 650.0

    # 공기비(λ) 운영 한계 [무차원] — 가스터빈 초희박 연소(2~4배 잉여 공기) 정상 대역.
    # 종전 가안(0.8~1.4)은 연료과농 영역으로 가스터빈 운전 원리에 위배 → 도메인 검토 후 전면 재설정.
    lambda_caution_lo: float = 2.0
    lambda_caution_hi: float = 3.5
    lambda_danger_lo: float = 1.5
    lambda_danger_hi: float = 4.0


# ============================================================
# 5. 시뮬 step 설정
# ============================================================
@dataclass(frozen=True)
class SimStepConfig:
    """시뮬레이션 step 단위 설정."""

    dt: float = 0.2               # step 시간 [초]. backend sim_dt_seconds와 일치시킬 것.
    physics_blend_ratio: float = 0.0  # NOx 블렌딩 비율 (0=ML만, 1=Zeldovich만) [추후 결정]


# ============================================================
# 6. 파생 피처 계산 상수
# [조사 필요] 실측 데이터 기반 캘리브레이션 필요
# ============================================================
@dataclass(frozen=True)
class FeatureConfig:
    """공기비·CO·효율 계산식 상수."""

    # compute_lambda
    base_lambda: float = 2.50          # 기준 운전점에서의 λ [무차원] — GE 7F/7FA 초희박 연소 정상 대역
    n2_correction: float = 0.0005      # N2 1단위 증가당 λ 보정 계수
    n2_scale: float = 1000.0           # N2 몰분율 환산 스케일 (delta_n2 계산용)

    # compute_co
    base_co: float = 12.0              # λ=1에서의 CO 베이스라인 [ppm, 가안]
    co_sensitivity: float = 80.0       # CO-λ 민감도 [(λ-1)^2 가중치]

    # compute_efficiency
    base_efficiency: float = 0.89      # 기준 발전 효율 [무차원]
    temp_sensitivity: float = 0.0001   # 온도 1°C 변동당 효율 변동 [1/°C]
    off_design_penalty: float = 0.02   # off-design 효율 페널티 계수

    # compute_o2_fraction
    o2_in_air: float = 0.21            # 공기 중 O2 몰분율 [물리상수, 대기 조성]

    # compute_n2_fraction
    n2_in_air: float = 0.79            # 공기 중 N2 몰분율 [물리상수, 대기 조성]


# ============================================================
# 7. Zeldovich 반응 상수
# [조사 필요] 합성가스(H2/CO) 적합값으로 교체 필요. 현재 메탄 문헌값 가안.
# ============================================================
@dataclass(frozen=True)
class ChemistryConfig:
    """Arrhenius 형태 Zeldovich 반응 상수."""

    pre_exponential_A: float = 1.8e8       # k1 pre-exponential factor [m³/(mol·s)]
    activation_energy_Ea: float = 318000.0  # k1 활성화 에너지 [J/mol, ~76 kcal/mol]
    scale_ppm: float = 1.0e-6              # ppm 스케일 정규화 계수 [조사 필요]


# ============================================================
# 전체 설정 묶음 — 런타임에서 단일 객체로 전달
# ============================================================
@dataclass(frozen=True)
class DTConfig:
    """디지털 트윈 전체 설정. 각 서브 설정을 필드로 보유."""

    operating_point: OperatingPoint = field(default_factory=OperatingPoint)
    initial_output: InitialOutput = field(default_factory=InitialOutput)
    time_constants: TimeConstants = field(default_factory=TimeConstants)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    sim_step: SimStepConfig = field(default_factory=SimStepConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    chemistry: ChemistryConfig = field(default_factory=ChemistryConfig)


DEFAULT_CONFIG = DTConfig()
