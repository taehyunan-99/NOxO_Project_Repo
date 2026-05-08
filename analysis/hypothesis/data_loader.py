"""
hypothesis/data_loader.py
=========================
가설 검증용 공통 데이터 로더.

production 모델(digital_twin/preprocess.py)과 달리,
모든 센서 컬럼을 로드해 가설 검증에 필요한 파생 피처를 만들 수 있도록 한다.

데이터 형식
-----------
- Row 0: TagName (컬럼명으로 사용)
- Row 1-4: Description, Units, Min, Max (skiprows=[1,2,3,4]로 제거)
- Row 5+: 1초 간격 실측값
- encoding='utf-8-sig' (BOM 제거)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─── 프로젝트 루트 기준 경로 ────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = _ROOT / "data"

TRAIN_FILE = DATA_DIR / "NOx_train_20250811_20250824.csv"
TEST_FILE  = DATA_DIR / "NOx_test_20250825.csv"

# ─── 공통 태그 이름 ─────────────────────────────────────────────────────────
TARGET_NOX   = "IGCC.DeNOX.AT_H1_901_PV"   # SCR 입구 NOx [ppm]
TARGET_DWATT = "IGCC.CC.G1.DWATT"           # 발전량 [MW]
TARGET_TTXM  = "IGCC.CC.G1.TTXM"           # 배기가스온도 [°C]

TARGETS = [TARGET_NOX, TARGET_DWATT, TARGET_TTXM]

# 가설 검증에 사용되는 핵심 컬럼 (출력이 목표이므로 타깃도 포함)
KEY_COLS = {
    "o2":    "IGCC.DeNOX.AIT_H1_902",       # SCR 입구 O2 [%]
    "dwatt": TARGET_DWATT,
    "nqj":   "IGCC.CC.G1.NQJ",              # N2 주입 유량
    "ttxm":  TARGET_TTXM,
    "vnpr_s": "IGCC.CC.G1.VNPR_S",          # NPR-S
    "vnpr_p": "IGCC.CC.G1.VNPR_P",          # NPR-P
    "nox":   TARGET_NOX,
}

# 1초 간격 기준 lag 스텝 수
LAG_1MIN  = 60
LAG_3MIN  = 180
LAG_5MIN  = 300
LAG_15MIN = 900
LAG_30MIN = 1800


# ─── 로더 ───────────────────────────────────────────────────────────────────

def load_raw(path: Path | str, nrows: Optional[int] = None) -> pd.DataFrame:
    """CSV를 원시 상태(모든 컬럼)로 읽어 반환.

    Parameters
    ----------
    path:
        CSV 파일 경로.
    nrows:
        읽을 최대 행 수. None이면 전체 로드.

    Returns
    -------
    pd.DataFrame
        인덱스 = 첫 번째 컬럼(날짜/시간), 값은 모두 float.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")

    df = pd.read_csv(
        path,
        header=0,
        skiprows=[1, 2, 3, 4],
        index_col=0,
        encoding="utf-8-sig",
        nrows=nrows,
    )
    before = len(df)
    df = df.apply(pd.to_numeric, errors="coerce")

    # 완전히 빈 컬럼 먼저 제거 (예: Column1, ttfr1 등 데이터 없는 컬럼)
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        df = df.drop(columns=all_nan_cols)
        print(f"[load_raw] 전체 NaN 컬럼 {len(all_nan_cols)}개 제거: {all_nan_cols}")

    df = df.dropna(how="all")           # 완전히 빈 행만 제거 (컬럼별 dropna는 각 스크립트에서)
    dropped = before - len(df)
    if dropped:
        print(f"[load_raw] 완전빈 행 {dropped}개 제거 → {len(df)}행")

    return df


def load_for_hypothesis(
    path: Path | str = TRAIN_FILE,
    cols: Optional[list[str]] = None,
    nrows: Optional[int] = None,
    sample_frac: float = 1.0,
) -> pd.DataFrame:
    """가설 검증용 데이터 로드.

    Parameters
    ----------
    path:
        CSV 파일 경로.
    cols:
        필요한 컬럼 목록. None이면 전체.
    nrows:
        읽을 최대 행 수.
    sample_frac:
        0 < sample_frac <= 1. 대용량 데이터 빠른 탐색 시 사용.

    Returns
    -------
    pd.DataFrame
        결측값 행이 제거된 데이터프레임.
    """
    df = load_raw(path, nrows=nrows)

    if cols:
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"필요한 컬럼이 없습니다: {sorted(missing)}")
        df = df[cols]

    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        print(f"[load_for_hypothesis] NaN 행 {dropped}개 제거 → {len(df)}행")

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).sort_index()
        print(f"[load_for_hypothesis] 샘플링 후 {len(df)}행")

    return df


def make_lag_features(
    df: pd.DataFrame,
    col: str,
    lags: list[int],
) -> pd.DataFrame:
    """시계열 lag 피처 생성 (1초 단위 데이터 기준).

    Parameters
    ----------
    df:
        원본 DataFrame.
    col:
        lag를 만들 컬럼명.
    lags:
        lag 스텝 수 리스트. 예: [60, 300] → 1분, 5분 선행.

    Returns
    -------
    pd.DataFrame
        lag 컬럼이 추가된 복사본. 컬럼명 예: '{col}_lag60s', '{col}_lag300s'.
    """
    df = df.copy()
    for lag in lags:
        if lag < 60:
            unit_label = f"{lag}s"
        else:
            mins = lag // 60
            unit_label = f"{mins}min"
        df[f"{col}_lag_{unit_label}"] = df[col].shift(lag)
    return df


def make_rolling_features(
    df: pd.DataFrame,
    col: str,
    windows: list[int],
) -> pd.DataFrame:
    """이동평균/이동표준편차 피처 생성.

    Parameters
    ----------
    df:
        원본 DataFrame.
    col:
        rolling을 적용할 컬럼명.
    windows:
        윈도우 크기(행 수) 리스트. 예: [300, 900] → 5분, 15분.

    Returns
    -------
    pd.DataFrame
        rolling_mean, rolling_std 컬럼이 추가된 복사본.
    """
    df = df.copy()
    for w in windows:
        if w < 60:
            unit_label = f"{w}s"
        else:
            mins = w // 60
            unit_label = f"{mins}min"
        df[f"{col}_roll_mean_{unit_label}"] = (
            df[col].rolling(window=w, min_periods=max(1, w // 2)).mean()
        )
        df[f"{col}_roll_std_{unit_label}"] = (
            df[col].rolling(window=w, min_periods=max(1, w // 2)).std()
        )
    return df


def train_test_split_temporal(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """시계열 순서를 유지한 train/test 분리 (shuffle 없음)."""
    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio는 0과 1 사이여야 합니다: {test_ratio}")
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def print_section(title: str) -> None:
    """터미널 섹션 구분선 출력."""
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
