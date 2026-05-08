"""
hypothesis/h4_temp_lag.py
===========================
가설 4: 배기가스 온도 장기 지연 검증
======================================

검증 목표
---------
배기가스 온도(TTXM)의 변화가 즉시 NOx에 반영되는지,
긴 열관성 지연을 거쳐 반영되는지 확인.

생성 피처 (1초 간격 기준)
--------------------------
- TTXM_diff_1min, diff_5min    : 1분/5분 차이 (변화율)
- TTXM_roll_mean_5min, 15min   : 이동평균 (열관성)
- TTXM_roll_std_5min, 15min    : 이동표준편차 (안정성)
- TTXM_lag_1min, 5min, 15min, 30min : 지연 온도

판단 기준
---------
- lag 피처 R² > 동시간 R² × 1.05 → 열관성 지연 유의미
- 최적 지연 > 15분 → TTXM lag 피처 필수

사용법
------
python -m analysis.hypothesis.h4_temp_lag
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from analysis.hypothesis.data_loader import (
    KEY_COLS,
    LAG_15MIN,
    LAG_1MIN,
    LAG_30MIN,
    LAG_5MIN,
    TARGETS,
    TRAIN_FILE,
    load_for_hypothesis,
    make_lag_features,
    make_rolling_features,
    print_section,
    train_test_split_temporal,
)

REPORT_DIR = _ROOT / "analysis" / "reports" / "hypothesis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = KEY_COLS["nox"]
TTXM_COL = KEY_COLS["ttxm"]

DIFF_STEPS   = [LAG_1MIN, LAG_5MIN]
ROLL_WINDOWS = [LAG_5MIN, LAG_15MIN]
LAG_STEPS    = [LAG_1MIN, LAG_5MIN, LAG_15MIN, LAG_30MIN]


def _single_r2(X_train: pd.Series, y_train: pd.Series,
               X_test: pd.Series, y_test: pd.Series) -> float:
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    model.fit(X_train.values.reshape(-1, 1), y_train)
    return r2_score(y_test, model.predict(X_test.values.reshape(-1, 1)))


def run(
    data_path: Path = TRAIN_FILE,
    nrows: int | None = 200_000,
    test_ratio: float = 0.2,
) -> dict:
    """가설 4 검증 실행."""
    print_section("가설 4: 배기가스 온도 장기 지연 검증")

    # ── 1. 데이터 로드 ───────────────────────────────────────────────
    needed = [TARGET, TTXM_COL]
    df_raw = load_for_hypothesis(data_path, cols=needed, nrows=nrows)
    print(f"\n데이터: {len(df_raw):,}행")

    # ── 2. 파생 피처 생성 ────────────────────────────────────────────
    df = df_raw.copy()

    # 변화량 (diff)
    for step in DIFF_STEPS:
        mins = step // 60
        df[f"{TTXM_COL}_diff_{mins}min"] = df[TTXM_COL].diff(step)

    # 이동평균 / 이동표준편차
    df = make_rolling_features(df, TTXM_COL, ROLL_WINDOWS)

    # lag
    df = make_lag_features(df, TTXM_COL, LAG_STEPS)

    df = df.dropna()
    print(f"피처 생성 후: {len(df):,}행")

    feat_cols = [c for c in df.columns if c != TARGET]

    # ── 3. 교차상관으로 최적 지연 탐색 ─────────────────────────────
    # 전체 시계열 사용 (최대 1800 스텝 = 30분 범위)
    nox_arr  = (df[TARGET] - df[TARGET].mean()).values
    temp_arr = (df[TTXM_COL] - df[TTXM_COL].mean()).values
    max_lag_steps = LAG_30MIN

    cross_corr = signal.correlate(nox_arr, temp_arr, mode="full")
    lags_arr   = signal.correlation_lags(len(nox_arr), len(temp_arr), mode="full")

    # 양의 lag만 (온도가 먼저, NOx가 나중)
    mid = len(cross_corr) // 2
    cc_positive   = cross_corr[mid:]
    lags_positive = lags_arr[mid:]

    best_lag_idx  = int(np.argmax(np.abs(cc_positive[:max_lag_steps])))
    best_lag_step = int(lags_positive[best_lag_idx])
    best_cc       = float(cc_positive[best_lag_idx])

    best_lag_min  = best_lag_step // 60
    print(f"\n  교차상관 최적 시차: {best_lag_step}초 ({best_lag_min}분), CC={best_cc:.2f}")

    # ── 4. 단일 피처 R² 비교 ─────────────────────────────────────────
    train_df, test_df = train_test_split_temporal(df, test_ratio)
    y_train = train_df[TARGET]
    y_test  = test_df[TARGET]

    results: dict[str, dict] = {}
    print("\n[단일 피처 R² 비교]")
    print(f"  {'피처':<40} {'R²':>8}  {'종류'}")
    print("  " + "-" * 60)

    for col in feat_cols:
        r2 = _single_r2(train_df[col], y_train, test_df[col], y_test)
        kind = (
            "lag" if "_lag_" in col
            else "diff" if "_diff_" in col
            else "roll_mean" if "_roll_mean_" in col
            else "roll_std" if "_roll_std_" in col
            else "동시간"
        )
        print(f"  {col:<40} {r2:>8.4f}  {kind}")
        results[col] = {"r2": r2, "kind": kind}

    # ── 5. 판정 ─────────────────────────────────────────────────────
    r2_current = results.get(TTXM_COL, {}).get("r2", 0)
    r2_lag_best = max(
        (v["r2"] for k, v in results.items() if v["kind"] == "lag"), default=0
    )
    best_lag_feat = max(
        ((k, v["r2"]) for k, v in results.items() if v["kind"] == "lag"),
        key=lambda x: x[1], default=("", 0)
    )[0]
    r2_diff_best  = max(
        (v["r2"] for k, v in results.items() if v["kind"] == "diff"), default=0
    )

    print_section("판정 결과")
    print(f"  TTXM 동시간 R²:     {r2_current:.4f}")
    print(f"  TTXM lag 최고 R²:   {r2_lag_best:.4f}  ({best_lag_feat})")
    print(f"  TTXM diff 최고 R²:  {r2_diff_best:.4f}")
    print(f"  교차상관 최적 지연: {best_lag_min}분")

    # 단일 피처로 예측 불가 케이스 (R² < 0)
    all_negative = all(v["r2"] < 0 for v in results.values())
    if all_negative:
        verdict = "SINGLE_FEATURE_INSUFFICIENT"
        print("\n  💡 단일 TTXM 파생 피처만으로는 NOx 예측 불가 (R² < 0)")
        print("     → TTXM은 다변수 모델에서만 유효한 피처")
        print("     → 다변수 LightGBM SHAP 분석으로 기여도 확인 권장")
    elif r2_lag_best > r2_current * 1.05:
        verdict = "LAG_EFFECTIVE"
        print("\n  ⚠️  열관성 지연 유의미 (lag > 동시간 +5%)")
        if best_lag_min >= 15:
            print(f"     → {best_lag_min}분 지연 — 장기 열관성 확인")
        print(f"     → 모델에 {best_lag_feat} 포함 권장")
    elif r2_diff_best > r2_current * 1.05:
        verdict = "DIFF_EFFECTIVE"
        print("\n  💡 온도 변화율(diff)이 더 중요")
        print("     → TTXM_diff 피처 포함 권장")
    else:
        verdict = "CURRENT_SUFFICIENT"
        print("\n  ✅ 동시간 TTXM으로 충분")
        print("     → 별도 lag 피처 불필요")

    # ── 6. 시각화 ───────────────────────────────────────────────────
    _plot(results, df, lags_positive, cc_positive, max_lag_steps, best_lag_step)

    return {
        "verdict": verdict,
        "r2_current": r2_current,
        "r2_lag_best": r2_lag_best,
        "r2_diff_best": r2_diff_best,
        "best_lag_min": best_lag_min,
        "best_lag_feat": best_lag_feat,
    }


def _plot(
    results: dict, df: pd.DataFrame,
    lags_positive: np.ndarray, cc_positive: np.ndarray,
    max_lag_steps: int, best_lag_step: int,
) -> None:
    feat_order = sorted(results.keys())
    r2_vals  = [results[k]["r2"] for k in feat_order]
    colors_map = {
        "lag":        "steelblue",
        "diff":       "green",
        "roll_mean":  "orange",
        "roll_std":   "purple",
        "동시간":     "gray",
    }
    bar_colors = [colors_map.get(results[k]["kind"], "gray") for k in feat_order]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1. 교차상관
    axes[0, 0].plot(lags_positive[:max_lag_steps], cc_positive[:max_lag_steps])
    axes[0, 0].axvline(best_lag_step, color="red", linestyle="--", linewidth=1.5,
                       label=f"최적 {best_lag_step}초 ({best_lag_step//60}분)")
    axes[0, 0].set_xlabel("시차 [초]")
    axes[0, 0].set_ylabel("교차상관")
    axes[0, 0].set_title("가설4: TTXM-NOx 교차상관")
    axes[0, 0].legend(fontsize=8)

    # 2. 피처별 R²
    axes[0, 1].barh(range(len(feat_order)), r2_vals, color=bar_colors)
    axes[0, 1].set_yticks(range(len(feat_order)))
    axes[0, 1].set_yticklabels(feat_order, fontsize=7)
    axes[0, 1].set_xlabel("R² Score")
    axes[0, 1].set_title("피처별 NOx 예측력")
    legend_patches = [
        plt.matplotlib.patches.Patch(color=c, label=k)
        for k, c in colors_map.items()
    ]
    axes[0, 1].legend(handles=legend_patches, fontsize=7)

    # 3. TTXM vs NOx 시계열 (앞 3000초)
    sample_n = min(3000, len(df))
    temp_s = df[TTXM_COL].iloc[:sample_n]
    nox_s  = df[TARGET].iloc[:sample_n]
    ax_twin = axes[1, 0].twinx()
    axes[1, 0].plot(temp_s.values, color="red",  alpha=0.7, label="TTXM", linewidth=0.8)
    ax_twin.plot(nox_s.values,    color="blue", alpha=0.7, label="NOx",  linewidth=0.8)
    axes[1, 0].set_ylabel("TTXM [°C]", color="red")
    ax_twin.set_ylabel("NOx [ppm]", color="blue")
    axes[1, 0].set_title("TTXM vs NOx 시계열 (앞 3000초)")
    axes[1, 0].set_xlabel("시간 [초]")

    # 4. NOx vs 동시간 TTXM 산점도
    sample_n = min(5000, len(df))
    s = df.sample(sample_n, random_state=42)
    corr_v = df[[TTXM_COL, TARGET]].corr().iloc[0, 1]
    axes[1, 1].scatter(s[TTXM_COL], s[TARGET], alpha=0.3, s=5, color="gray")
    axes[1, 1].set_xlabel("TTXM [°C]")
    axes[1, 1].set_ylabel("NOx [ppm]")
    axes[1, 1].set_title(f"NOx vs TTXM  (r={corr_v:.3f})")

    plt.tight_layout()
    save_path = REPORT_DIR / "h4_temp_lag.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📈 그래프 저장: {save_path}")


if __name__ == "__main__":
    run()
