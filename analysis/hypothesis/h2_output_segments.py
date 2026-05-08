"""
hypothesis/h2_output_segments.py
==================================
가설 2: 발전 출력 구간별 NOx 생성 메커니즘 차이
==================================================

검증 목표
---------
하나의 전체 LightGBM 모델보다 DWATT(발전량) 구간별 분리 모델이
NOx 예측을 더 안정적으로 수행하는지 비교.

출력 구간 (MW)
--------------
- Low   : DWATT < 140
- Mid1  : 140 ≤ DWATT < 160
- Mid2  : 160 ≤ DWATT < 180
- High1 : 180 ≤ DWATT < 200
- High2 : 200 ≤ DWATT

판단 기준
---------
- 구간별 R² 평균 > 전체 R² × 1.05 → 구간별 모델 또는 hinge 피처 채택
- 구간별 개선 < 5%               → 통합 모델 유지

사용법
------
python -m analysis.hypothesis.h2_output_segments
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from analysis.hypothesis.data_loader import (
    KEY_COLS,
    TARGETS,
    TRAIN_FILE,
    load_for_hypothesis,
    print_section,
    train_test_split_temporal,
)

REPORT_DIR = _ROOT / "analysis" / "reports" / "hypothesis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET    = KEY_COLS["nox"]
DWATT_COL = KEY_COLS["dwatt"]

# 출력 구간 (MW)
BINS   = [0, 140, 160, 180, 200, float("inf")]
LABELS = ["<140", "140-160", "160-180", "180-200", "≥200"]

_LGBM_PARAMS = dict(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


def _fit_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label: str,
) -> dict:
    model = LGBMRegressor(**_LGBM_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"    {label:<25}  R²={r2:+.4f}  MAE={mae:.4f}")
    return {"r2": r2, "mae": mae, "y_pred": y_pred, "model": model,
            "n_test": len(y_test)}


def run(
    data_path: Path = TRAIN_FILE,
    nrows: int | None = 200_000,
    test_ratio: float = 0.2,
) -> dict:
    """가설 2 검증 실행."""
    print_section("가설 2: 발전 출력 구간별 NOx 메커니즘 차이")

    # ── 1. 데이터 로드 ───────────────────────────────────────────────
    df = load_for_hypothesis(data_path, nrows=nrows)
    if TARGET not in df.columns or DWATT_COL not in df.columns:
        raise ValueError(f"필수 컬럼 없음: {TARGET}, {DWATT_COL}")

    feature_cols = [c for c in df.columns if c not in TARGETS]
    print(f"\n데이터: {len(df):,}행  |  피처: {len(feature_cols)}개")

    # ── 2. 전체 통합 모델 ────────────────────────────────────────────
    train_df, test_df = train_test_split_temporal(df, test_ratio)
    X_train_all = train_df[feature_cols]
    y_train_all = train_df[TARGET]
    X_test_all  = test_df[feature_cols]
    y_test_all  = test_df[TARGET]

    print("\n[전체 통합 모델]")
    global_result = _fit_lgbm(X_train_all, y_train_all,
                               X_test_all,  y_test_all, "전체 통합")

    # ── 3. 구간별 분석 & 모델 ────────────────────────────────────────
    print("\n[구간별 모델 비교]")
    seg_results: dict[str, dict] = {}

    for label, (lo, hi) in zip(LABELS, zip(BINS[:-1], BINS[1:])):
        mask_train = (train_df[DWATT_COL] >= lo) & (train_df[DWATT_COL] < hi)
        mask_test  = (test_df[DWATT_COL]  >= lo) & (test_df[DWATT_COL]  < hi)

        seg_train = train_df[mask_train]
        seg_test  = test_df[mask_test]

        n_train = len(seg_train)
        n_test  = len(seg_test)
        nox_mean = seg_test[TARGET].mean() if n_test > 0 else np.nan
        nox_std  = seg_test[TARGET].std()  if n_test > 0 else np.nan

        print(f"\n  [{label} MW]  train={n_train:,}  test={n_test:,}")
        print(f"    NOx 평균={nox_mean:.4f}  std={nox_std:.4f}")

        if n_train < 100 or n_test < 30:
            print("    ⚠️  데이터 부족 — 구간 건너뜀")
            seg_results[label] = {
                "r2": np.nan, "mae": np.nan,
                "n_test": n_test, "nox_mean": nox_mean, "nox_std": nox_std,
            }
            continue

        # 구간별 모델
        result = _fit_lgbm(
            seg_train[feature_cols], seg_train[TARGET],
            seg_test[feature_cols],  seg_test[TARGET],
            f"구간({label}MW)",
        )
        result["nox_mean"] = nox_mean
        result["nox_std"]  = nox_std
        seg_results[label] = result

        # 구간 내 피처 상관 상위 5개
        corr_s = seg_train[feature_cols + [TARGET]].corr()[TARGET].drop(TARGET)
        top5 = corr_s.abs().sort_values(ascending=False).head(5)
        print(f"    상위 상관 피처: {', '.join(f'{c}({v:.2f})' for c, v in top5.items())}")

    # ── 4. 판정 ─────────────────────────────────────────────────────
    valid_r2 = [v["r2"] for v in seg_results.values() if not np.isnan(v.get("r2", np.nan))]
    avg_seg_r2 = np.mean(valid_r2) if valid_r2 else np.nan
    global_r2  = global_result["r2"]
    improvement = (avg_seg_r2 - global_r2) / abs(global_r2) * 100 if global_r2 != 0 else 0

    print_section("판정 결과")
    print(f"  전체 통합 모델 R²: {global_r2:.4f}")
    print(f"  구간별 모델 평균 R²: {avg_seg_r2:.4f}  (개선: {improvement:+.2f}%)")

    if improvement > 5:
        verdict = "SEGMENT_EFFECTIVE"
        print("\n  ✅ 구간별 모델이 유효 (>5% 개선)")
        print("     → DWATT hinge 피처 또는 구간별 분리 모델 권장")
    elif improvement > 0:
        verdict = "SEGMENT_MARGINAL"
        print("\n  ⚠️  구간별 개선 미미 (<5%)")
        print("     → 모델 복잡도 고려 후 결정")
    else:
        verdict = "GLOBAL_BETTER"
        print("\n  💡 통합 모델이 안정적")
        print("     → DWATT 구간 분리 불필요")

    # ── 5. 시각화 ───────────────────────────────────────────────────
    _plot(global_result, seg_results, df)

    return {
        "verdict": verdict,
        "global_r2": global_r2,
        "avg_segment_r2": avg_seg_r2,
        "improvement_pct": improvement,
        "segment_results": {k: {"r2": v.get("r2"), "mae": v.get("mae")}
                            for k, v in seg_results.items()},
    }


def _plot(global_result: dict, seg_results: dict, df: pd.DataFrame) -> None:
    valid_segs = {k: v for k, v in seg_results.items()
                  if not np.isnan(v.get("r2", np.nan))}
    labels = list(valid_segs.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1. 구간별 R² vs 전체 모델
    r2_seg_vals = [valid_segs[k]["r2"] for k in labels]
    axes[0, 0].bar(labels, r2_seg_vals, color="steelblue", label="구간별 모델")
    axes[0, 0].axhline(global_result["r2"], color="red", linestyle="--",
                       linewidth=1.5, label=f"전체 모델 (R²={global_result['r2']:.3f})")
    axes[0, 0].set_ylabel("R² Score")
    axes[0, 0].set_title("가설2: 구간별 R² vs 전체 모델")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylim([0, 1])

    # 2. 구간별 NOx 평균 ± std
    nox_means = [valid_segs[k]["nox_mean"] for k in labels]
    nox_stds  = [valid_segs[k]["nox_std"] for k in labels]
    axes[0, 1].bar(labels, nox_means, yerr=nox_stds, capsize=5, color="orange", alpha=0.8)
    axes[0, 1].set_ylabel("NOx [ppm]")
    axes[0, 1].set_title("구간별 NOx 평균 ± std")

    # 3. DWATT 분포
    axes[1, 0].hist(df[DWATT_COL].dropna(), bins=60, color="green", alpha=0.7)
    for b in BINS[1:-1]:
        axes[1, 0].axvline(b, color="red", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("발전 출력 [MW]")
    axes[1, 0].set_ylabel("빈도")
    axes[1, 0].set_title("DWATT 분포 (구간 경계)")

    # 4. 구간별 MAE 비교
    mae_vals = [valid_segs[k]["mae"] for k in labels]
    axes[1, 1].bar(labels, mae_vals, color="purple", alpha=0.7)
    axes[1, 1].axhline(global_result["mae"], color="red", linestyle="--",
                       linewidth=1.5, label=f"전체 모델 MAE={global_result['mae']:.4f}")
    axes[1, 1].set_ylabel("MAE [ppm]")
    axes[1, 1].set_title("구간별 MAE 비교")
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    save_path = REPORT_DIR / "h2_output_segments.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📈 그래프 저장: {save_path}")


if __name__ == "__main__":
    run()
