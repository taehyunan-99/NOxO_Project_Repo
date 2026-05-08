"""
hypothesis/h1_o2_leakage.py
============================
가설 1: SCR 입구 O2 준누수 검증
================================

검증 목표
---------
O2(AIT_H1_902)가 공정 상태를 반영하는 유효 피처인지,
아니면 NOx(AT_H1_901_PV)와 계측/보정 계통으로 연결된 준누수 피처인지 확인.

4가지 모델을 비교한다:
  1. O2 포함 전체 피처 (Ridge)
  2. O2 제외 전체 피처 (Ridge)
  3. O2 단독 (LinearRegression)
  4. O2 lag 60초, 300초 (선행 O2만 사용)

판단 기준
---------
- O2 단독 R² > 0.8  → 준누수 가능성 매우 높음
- O2 단독 R² 0.5~0.8 → 의심 수준
- O2 단독 R² < 0.5  → 공정 피처로 사용 가능

사용법
------
python -m analysis.hypothesis.h1_o2_leakage
  또는
python analysis/hypothesis/h1_o2_leakage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경 대응
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 프로젝트 루트를 sys.path에 추가 (직접 실행 시)
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from analysis.hypothesis.data_loader import (
    KEY_COLS,
    LAG_1MIN,
    LAG_5MIN,
    TARGETS,
    TRAIN_FILE,
    load_for_hypothesis,
    make_lag_features,
    print_section,
    train_test_split_temporal,
)

# 결과 저장 경로
REPORT_DIR = _ROOT / "analysis" / "reports" / "hypothesis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 가설 1에 사용할 컬럼
O2_COL    = KEY_COLS["o2"]      # AIT_H1_902
TARGET    = KEY_COLS["nox"]     # AT_H1_901_PV
DWATT_COL = KEY_COLS["dwatt"]
NQJ_COL   = KEY_COLS["nqj"]
TTXM_COL  = KEY_COLS["ttxm"]


def _fit_ridge(X_train: pd.DataFrame, y_train: pd.Series,
               X_test: pd.DataFrame, y_test: pd.Series,
               label: str) -> dict:
    """Ridge 회귀 학습 및 평가 (StandardScaler 포함)."""
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"  {label:<35} R²={r2:+.4f}  MAE={mae:.4f}")
    return {"label": label, "r2": r2, "mae": mae,
            "y_pred": y_pred, "model": model}


def run(
    data_path: Path = TRAIN_FILE,
    nrows: int | None = 200_000,
    sample_frac: float = 1.0,
    test_ratio: float = 0.2,
) -> dict:
    """가설 1 검증 실행.

    Parameters
    ----------
    data_path:
        학습 CSV 경로.
    nrows:
        읽을 행 수 (기본 200,000행 — 메모리 절약).
    sample_frac:
        추가 샘플링 비율.
    test_ratio:
        시계열 분리 비율.

    Returns
    -------
    dict
        각 모델의 R², MAE, 판정 결과.
    """
    print_section("가설 1: SCR 입구 O2 준누수 검증")

    # ── 1. 데이터 로드 ───────────────────────────────────────────────
    # 필요한 컬럼: NOx 타깃 + O2 + 나머지 주요 변수
    # 모든 컬럼을 가져온 뒤 dropna 적용
    df_raw = load_for_hypothesis(data_path, nrows=nrows, sample_frac=sample_frac)

    required = [TARGET, O2_COL, DWATT_COL, NQJ_COL, TTXM_COL]
    missing = set(required) - set(df_raw.columns)
    if missing:
        raise ValueError(f"필수 컬럼 없음: {sorted(missing)}")

    # feature 컬럼 = NOx 제외, 모든 수치형 컬럼
    all_feature_cols = [c for c in df_raw.columns if c not in TARGETS]
    df = df_raw[all_feature_cols + [TARGET]].copy()

    # ── 2. O2 lag 피처 추가 ─────────────────────────────────────────
    df = make_lag_features(df, O2_COL, [LAG_1MIN, LAG_5MIN])
    df = df.dropna()

    lag1_col = f"{O2_COL}_lag_1min"
    lag5_col = f"{O2_COL}_lag_5min"

    # ── 3. train/test 분리 ──────────────────────────────────────────
    train_df, test_df = train_test_split_temporal(df, test_ratio)

    y_train = train_df[TARGET]
    y_test  = test_df[TARGET]

    feat_all        = [c for c in all_feature_cols if c in df.columns]
    feat_no_o2      = [c for c in feat_all if c != O2_COL]
    feat_o2_only    = [O2_COL]
    feat_o2_lag     = [lag1_col, lag5_col]

    print(f"\n데이터: {len(df):,}행  |  train={len(train_df):,}  test={len(test_df):,}")
    print(f"전체 피처 수: {len(feat_all)}  |  O2 제외: {len(feat_no_o2)}")

    # ── 4. 모델 비교 ────────────────────────────────────────────────
    print("\n[Ridge 회귀 성능 비교]")
    results = {}
    for name, feats in [
        ("O2 포함 전체",   feat_all),
        ("O2 제외 전체",   feat_no_o2),
        ("O2 단독",        feat_o2_only),
        ("O2 lag (1min, 5min)", feat_o2_lag),
    ]:
        r = _fit_ridge(
            train_df[feats], y_train,
            test_df[feats],  y_test,
            name,
        )
        results[name] = r

    # ── 5. 판정 ─────────────────────────────────────────────────────
    r2_o2_only = results["O2 단독"]["r2"]
    r2_with_o2 = results["O2 포함 전체"]["r2"]
    r2_without  = results["O2 제외 전체"]["r2"]
    delta_r2 = r2_with_o2 - r2_without

    print_section("판정 결과")
    print(f"  O2 단독 R²: {r2_o2_only:.4f}")
    print(f"  O2 포함 vs 제외 R² 차이: {delta_r2:+.4f}")

    if r2_o2_only > 0.8:
        verdict = "LEAKAGE_HIGH"
        print("\n  ⚠️  O2 준누수 가능성 매우 높음!")
        print("     → O2는 계측/보정 계통 연결 피처일 수 있음")
        print("     → 예측 모델과 해석 모델 분리 권장")
    elif r2_o2_only > 0.5:
        verdict = "LEAKAGE_MEDIUM"
        print("\n  ⚠️  O2 준누수 의심 (중간 수준)")
        print("     → O2 포함/제외 모델 모두 구축 후 비교")
    else:
        verdict = "LEAKAGE_LOW"
        print("\n  ✅ O2 준누수 가능성 낮음")
        print("     → 공정 상태를 반영하는 유효 피처")

    if delta_r2 > 0.1:
        print(f"\n  💡 O2가 전체 성능의 {delta_r2*100:.1f}%p 담당")
    else:
        print(f"\n  💡 O2 제외 시 성능 저하 {delta_r2*100:.1f}%p (미미)")

    # ── 6. 시각화 ───────────────────────────────────────────────────
    _plot(results, df, r2_o2_only)

    return {
        "verdict": verdict,
        "r2_o2_only": r2_o2_only,
        "r2_with_o2": r2_with_o2,
        "r2_without_o2": r2_without,
        "delta_r2": delta_r2,
    }


def _plot(results: dict, df: pd.DataFrame, r2_o2_only: float) -> None:
    """결과 시각화 저장."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. R² 비교
    labels = list(results.keys())
    r2_vals = [results[k]["r2"] for k in labels]
    colors = ["steelblue", "orange", "crimson", "purple"]
    axes[0].bar(range(len(labels)), r2_vals, color=colors[:len(labels)])
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axes[0].set_ylabel("R² Score")
    axes[0].set_title("가설1 모델별 R² 비교")
    axes[0].set_ylim([0, 1])
    for i, v in enumerate(r2_vals):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # 2. NOx vs O2 산점도
    sample_n = min(5000, len(df))
    s = df.sample(sample_n, random_state=42)
    o2_col = KEY_COLS["o2"]
    nox_col = KEY_COLS["nox"]
    if o2_col in df.columns and nox_col in df.columns:
        corr_val = df[[o2_col, nox_col]].corr().iloc[0, 1]
        axes[1].scatter(s[o2_col], s[nox_col], alpha=0.3, s=5, color="steelblue")
        axes[1].set_xlabel("O2 (AIT_H1_902)")
        axes[1].set_ylabel("NOx (AT_H1_901_PV)")
        axes[1].set_title(f"NOx vs O2  (r={corr_val:.3f})")

    # 3. O2 단독 모델 잔차
    y_pred_o2 = results["O2 단독"]["y_pred"]
    # y_test가 여기서 없으므로 test split 재현
    n_test = len(y_pred_o2)
    y_true_tail = df[nox_col].iloc[-n_test:].values
    residuals = y_true_tail - y_pred_o2
    axes[2].scatter(y_pred_o2, residuals, alpha=0.3, s=5, color="crimson")
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_xlabel("O2 단독 예측값")
    axes[2].set_ylabel("잔차")
    axes[2].set_title(f"O2 단독 잔차  (R²={r2_o2_only:.3f})")

    plt.tight_layout()
    save_path = REPORT_DIR / "h1_o2_leakage.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📈 그래프 저장: {save_path}")


if __name__ == "__main__":
    run()
