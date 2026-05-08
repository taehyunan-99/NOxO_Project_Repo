"""
hypothesis/h5_npr_nonlinear.py
================================
가설 5: NPR 비선형성 및 상호작용 검증
========================================

검증 목표
---------
VNPR_P, VNPR_S(노즐 압력비)가 NOx와 단순 선형 관계인지,
특정 임계값 이후 비선형적으로 달라지는지,
NQJ/DWATT와 상호작용이 있는지 확인.

생성 피처
----------
- NPR_avg = (VNPR_P + VNPR_S) / 2
- NPR_gap = VNPR_P - VNPR_S
- NPR_hinge = max(0, NPR_avg - median)       (비선형 임계)
- NPR_x_NQJ = NPR_avg × NQJ                 (상호작용: N2 희석)
- NPR_x_DWATT = NPR_avg × DWATT             (상호작용: 출력 구간)
- NPR_x_csgv  = NPR_avg × csgv              (상호작용: 압축기 공기)

판단 기준
---------
- 다항식(2차) R² > 선형 R² × 1.1 → 비선형성 유의미
- NPR_x_NQJ R² > NPR_avg R²      → 상호작용 효과 유의미

사용법
------
python -m analysis.hypothesis.h5_npr_nonlinear
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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
VNPR_S    = KEY_COLS["vnpr_s"]
VNPR_P    = KEY_COLS["vnpr_p"]
NQJ_COL   = KEY_COLS["nqj"]
DWATT_COL = KEY_COLS["dwatt"]
CSGV_COL  = "IGCC.CC.G1.csgv"   # 압축기 스테이터 개도


def _single_r2(X_train: pd.Series | pd.DataFrame,
               y_train: pd.Series,
               X_test: pd.Series | pd.DataFrame,
               y_test: pd.Series) -> float:
    if isinstance(X_train, pd.Series):
        X_train = X_train.values.reshape(-1, 1)
        X_test  = X_test.values.reshape(-1, 1)
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    model.fit(X_train, y_train)
    return r2_score(y_test, model.predict(X_test))


def run(
    data_path: Path = TRAIN_FILE,
    nrows: int | None = 200_000,
    test_ratio: float = 0.2,
) -> dict:
    """가설 5 검증 실행."""
    print_section("가설 5: NPR 비선형성 및 상호작용 검증")

    # ── 1. 데이터 로드 ───────────────────────────────────────────────
    base_cols = [TARGET, VNPR_S, VNPR_P, NQJ_COL, DWATT_COL]
    df_raw = load_for_hypothesis(data_path, nrows=nrows)
    if CSGV_COL in df_raw.columns:
        base_cols.append(CSGV_COL)

    # 필요한 컬럼만 선택
    avail = [c for c in base_cols if c in df_raw.columns]
    df = df_raw[avail].dropna().copy()
    print(f"\n데이터: {len(df):,}행  |  사용 컬럼: {avail}")

    # ── 2. 파생 피처 생성 ────────────────────────────────────────────
    df["NPR_avg"]  = (df[VNPR_S] + df[VNPR_P]) / 2
    df["NPR_gap"]  = df[VNPR_P] - df[VNPR_S]

    npr_threshold  = float(df["NPR_avg"].median())
    df["NPR_hinge"] = np.maximum(0, df["NPR_avg"] - npr_threshold)
    print(f"  NPR_hinge 기준값(중간값): {npr_threshold:.4f}")

    df["NPR_x_NQJ"]   = df["NPR_avg"] * df[NQJ_COL]
    df["NPR_x_DWATT"] = df["NPR_avg"] * df[DWATT_COL]

    if CSGV_COL in df.columns:
        df["NPR_x_csgv"] = df["NPR_avg"] * df[CSGV_COL]

    # ── 3. train/test 분리 ──────────────────────────────────────────
    train_df, test_df = train_test_split_temporal(df, test_ratio)
    y_train = train_df[TARGET]
    y_test  = test_df[TARGET]

    derived_cols = [
        "NPR_avg", "NPR_gap", "NPR_hinge",
        "NPR_x_NQJ", "NPR_x_DWATT",
    ]
    if CSGV_COL in df.columns:
        derived_cols.append("NPR_x_csgv")

    # ── 4. 단일 피처 R² ──────────────────────────────────────────────
    results: dict[str, dict] = {}
    print("\n[NPR 파생 피처별 R² 비교]")
    print(f"  {'피처':<25} {'R²':>8}  {'종류'}")
    print("  " + "-" * 50)

    for col in derived_cols:
        r2 = _single_r2(train_df[col], y_train, test_df[col], y_test)
        kind = (
            "상호작용" if "x_" in col
            else "hinge(비선형)" if "hinge" in col
            else "기본"
        )
        print(f"  {col:<25} {r2:>8.4f}  {kind}")
        results[col] = {"r2": r2, "kind": kind}

    # ── 5. 선형 vs 다항식(2차) 비교 ─────────────────────────────────
    print("\n[NPR_avg: 선형 vs 다항식(2차) 비교]")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_train = poly.fit_transform(train_df[["NPR_avg"]])
    X_poly_test  = poly.transform(test_df[["NPR_avg"]])

    r2_linear = _single_r2(train_df["NPR_avg"], y_train,
                            test_df["NPR_avg"],  y_test)
    model_poly = Ridge(alpha=1.0)
    model_poly.fit(X_poly_train, y_train)
    r2_poly = r2_score(y_test, model_poly.predict(X_poly_test))
    nonlinear_gain = (r2_poly - r2_linear) / abs(r2_linear) * 100 if r2_linear != 0 else 0

    print(f"  NPR_avg 선형:   R²={r2_linear:.4f}")
    print(f"  NPR_avg 다항식: R²={r2_poly:.4f}  (개선: {nonlinear_gain:+.2f}%)")

    results["NPR_avg_linear"] = {"r2": r2_linear, "kind": "선형"}
    results["NPR_avg_poly2"]  = {"r2": r2_poly,   "kind": "다항식"}

    # ── 6. 판정 ─────────────────────────────────────────────────────
    r2_interaction_best = max(
        (v["r2"] for k, v in results.items() if v["kind"] == "상호작용"), default=0
    )
    best_interaction = max(
        ((k, v["r2"]) for k, v in results.items() if v["kind"] == "상호작용"),
        key=lambda x: x[1], default=("", 0)
    )[0]

    print_section("판정 결과")
    print(f"  NPR 선형 R²:    {r2_linear:.4f}")
    print(f"  NPR 다항식 R²:  {r2_poly:.4f}  (개선 {nonlinear_gain:+.2f}%)")
    print(f"  NPR 최고 상호작용 R²: {r2_interaction_best:.4f}  ({best_interaction})")

    verdicts = []
    if r2_poly > r2_linear * 1.1:
        verdicts.append("NONLINEAR")
        print("\n  ⚠️  NPR 비선형성 확인 (다항식 > 선형 × 1.1)")
        print("     → hinge 피처 또는 2차 항 추가 권장")
    else:
        verdicts.append("LINEAR")
        print("\n  ✅ NPR은 선형 관계로 충분")

    if r2_interaction_best > r2_linear * 1.1:
        verdicts.append("INTERACTION_EFFECTIVE")
        print(f"\n  💡 상호작용 피처 효과 유의미: {best_interaction}")
        print("     → 상호작용 피처 추가 권장")
    else:
        verdicts.append("INTERACTION_MARGINAL")
        print("\n  💡 상호작용 피처 효과 미미")

    # ── 7. 시각화 ───────────────────────────────────────────────────
    _plot(results, df, test_df, y_test, r2_linear, r2_poly,
          model_poly, poly, npr_threshold)

    return {
        "verdicts": verdicts,
        "r2_linear": r2_linear,
        "r2_poly": r2_poly,
        "nonlinear_gain_pct": nonlinear_gain,
        "r2_interaction_best": r2_interaction_best,
        "best_interaction": best_interaction,
    }


def _plot(
    results: dict,
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_test: pd.Series,
    r2_linear: float,
    r2_poly: float,
    model_poly,
    poly,
    npr_threshold: float,
) -> None:
    derived_keys = [k for k in results if k not in ("NPR_avg_linear", "NPR_avg_poly2")]
    r2_vals_d = [results[k]["r2"] for k in derived_keys]
    colors_map = {"기본": "steelblue", "hinge(비선형)": "orange", "상호작용": "crimson",
                  "선형": "gray", "다항식": "purple"}
    bar_colors_d = [colors_map.get(results[k]["kind"], "gray") for k in derived_keys]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1. 파생 피처별 R²
    axes[0, 0].barh(range(len(derived_keys)), r2_vals_d, color=bar_colors_d)
    axes[0, 0].set_yticks(range(len(derived_keys)))
    axes[0, 0].set_yticklabels(derived_keys, fontsize=8)
    axes[0, 0].set_xlabel("R² Score")
    axes[0, 0].set_title("가설5: NPR 파생 피처별 R²")
    legend_patches = [
        plt.matplotlib.patches.Patch(color=c, label=k)
        for k, c in colors_map.items()
    ]
    axes[0, 0].legend(handles=legend_patches, fontsize=7)

    # 2. NPR_avg vs NOx: 선형 vs 다항식
    sample_n = min(5000, len(df))
    s = df.sample(sample_n, random_state=42)
    npr_range = np.linspace(df["NPR_avg"].min(), df["NPR_avg"].max(), 200)

    from sklearn.linear_model import Ridge as _Ridge
    m_lin = _Ridge(alpha=1.0)
    m_lin.fit(test_df[["NPR_avg"]], y_test)

    pred_lin  = m_lin.predict(npr_range.reshape(-1, 1))
    pred_poly = model_poly.predict(poly.transform(npr_range.reshape(-1, 1)))

    axes[0, 1].scatter(s["NPR_avg"], s[TARGET], alpha=0.3, s=5, color="gray", label="실측")
    axes[0, 1].plot(npr_range, pred_lin,  color="steelblue", linewidth=2,
                    label=f"선형 (R²={r2_linear:.3f})")
    axes[0, 1].plot(npr_range, pred_poly, color="red", linewidth=2, linestyle="--",
                    label=f"2차 (R²={r2_poly:.3f})")
    axes[0, 1].axvline(npr_threshold, color="orange", linestyle=":", linewidth=1.5,
                       label=f"hinge={npr_threshold:.2f}")
    axes[0, 1].set_xlabel("NPR_avg")
    axes[0, 1].set_ylabel("NOx [ppm]")
    axes[0, 1].set_title("NPR vs NOx: 선형 vs 다항식")
    axes[0, 1].legend(fontsize=7)

    # 3. NPR_avg 분포
    axes[1, 0].hist(df["NPR_avg"].dropna(), bins=60, color="steelblue", alpha=0.7)
    axes[1, 0].axvline(npr_threshold, color="red", linestyle="--", linewidth=2,
                       label=f"hinge 기준 ({npr_threshold:.2f})")
    axes[1, 0].set_xlabel("NPR_avg")
    axes[1, 0].set_ylabel("빈도")
    axes[1, 0].set_title("NPR_avg 분포")
    axes[1, 0].legend(fontsize=8)

    # 4. NPR_x_NQJ vs NOx
    if "NPR_x_NQJ" in df.columns:
        s = df.sample(sample_n, random_state=42)
        r2_inter = results.get("NPR_x_NQJ", {}).get("r2", np.nan)
        axes[1, 1].scatter(s["NPR_x_NQJ"], s[TARGET], alpha=0.3, s=5, color="crimson")
        axes[1, 1].set_xlabel("NPR × NQJ (상호작용)")
        axes[1, 1].set_ylabel("NOx [ppm]")
        axes[1, 1].set_title(f"상호작용 피처 vs NOx  (R²={r2_inter:.3f})")

    plt.tight_layout()
    save_path = REPORT_DIR / "h5_npr_nonlinear.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📈 그래프 저장: {save_path}")


if __name__ == "__main__":
    run()
