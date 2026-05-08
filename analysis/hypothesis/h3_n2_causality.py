"""
hypothesis/h3_n2_causality.py
===============================
가설 3: N2 주입 역인과 검증
==============================

검증 목표
---------
NQJ(N2 주입 유량)가 NOx를 낮추는 원인인지,
NOx 상승에 대응한 제어 반응(역인과)인지 구분.

검증 방법
---------
- NQJ_lag_1min, 3min, 5min: 선행 N2 (t-lag) — 원인 방향
- NQJ_lead_1min, 3min:      미래 N2 (t+lead) — 역인과 방향
- 각각 단독 Ridge로 NOx를 예측하고 R²를 비교

판단 기준
---------
- R²(lead) > R²(lag) × 1.2 → 강한 역인과
- R²(lag) ≥ R²(current)    → N2는 원인 효과

사용법
------
python -m analysis.hypothesis.h3_n2_causality
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from analysis.hypothesis.data_loader import (
    KEY_COLS,
    LAG_1MIN,
    LAG_3MIN,
    LAG_5MIN,
    TARGETS,
    TRAIN_FILE,
    load_for_hypothesis,
    make_lag_features,
    print_section,
    train_test_split_temporal,
)

REPORT_DIR = _ROOT / "analysis" / "reports" / "hypothesis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET  = KEY_COLS["nox"]
NQJ_COL = KEY_COLS["nqj"]

# lag/lead 스텝 (1초 간격 기준)
LAGS  = [LAG_1MIN, LAG_3MIN, LAG_5MIN]    # 선행 (원인 후보)
LEADS = [LAG_1MIN, LAG_3MIN]               # 미래 (역인과 검증)


def _single_r2(X_train: pd.Series, y_train: pd.Series,
               X_test: pd.Series, y_test: pd.Series) -> float:
    """단일 피처 Ridge R² (StandardScaler 포함)."""
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    model.fit(X_train.values.reshape(-1, 1), y_train)
    y_pred = model.predict(X_test.values.reshape(-1, 1))
    return r2_score(y_test, y_pred)


def run(
    data_path: Path = TRAIN_FILE,
    nrows: int | None = 200_000,
    test_ratio: float = 0.2,
) -> dict:
    """가설 3 검증 실행."""
    print_section("가설 3: N2 주입 역인과 검증")

    # ── 1. 데이터 로드 ───────────────────────────────────────────────
    needed = [TARGET, NQJ_COL]
    df = load_for_hypothesis(data_path, cols=needed, nrows=nrows)
    print(f"\n데이터: {len(df):,}행")

    # ── 2. lag/lead 피처 생성 ────────────────────────────────────────
    # 선행 NQJ: t-lag (NQJ가 먼저 바뀜 → NOx에 영향)
    for lag in LAGS:
        mins = lag // 60
        df[f"NQJ_lag_{mins}min"] = df[NQJ_COL].shift(lag)

    # 미래 NQJ: t+lead (NOx 상승 후 NQJ가 대응)
    for lead in LEADS:
        mins = lead // 60
        df[f"NQJ_lead_{mins}min"] = df[NQJ_COL].shift(-lead)

    df = df.dropna()
    print(f"lag/lead 생성 후: {len(df):,}행")

    # ── 3. train/test 분리 ──────────────────────────────────────────
    train_df, test_df = train_test_split_temporal(df, test_ratio)
    y_train = train_df[TARGET]
    y_test  = test_df[TARGET]

    # ── 4. 단일 피처 R² 비교 ─────────────────────────────────────────
    feat_cols = [c for c in df.columns if c != TARGET]
    results: dict[str, dict] = {}

    print("\n[단일 피처 R² 비교]")
    print(f"  {'피처':<30} {'R²':>8}  {'NOx 상관':>10}  {'의미'}")
    print("  " + "-" * 65)

    for col in feat_cols:
        r2  = _single_r2(train_df[col], y_train, test_df[col], y_test)
        corr, pval = stats.pearsonr(df[col], df[TARGET])
        kind = (
            "lag (선행-원인)" if "lag" in col
            else "lead (미래-역인과)" if "lead" in col
            else "동시간"
        )
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"  {col:<30} {r2:>8.4f}  {corr:>10.4f}{sig:<3}  {kind}")
        results[col] = {"r2": r2, "corr": corr, "pval": pval, "kind": kind}

    # ── 5. 판정 ─────────────────────────────────────────────────────
    r2_current = results.get(NQJ_COL, {}).get("r2", 0)
    r2_lag_best  = max((v["r2"] for k, v in results.items() if "lag" in k), default=0)
    r2_lead_best = max((v["r2"] for k, v in results.items() if "lead" in k), default=0)

    print_section("판정 결과")
    print(f"  동시간 NQJ R²:    {r2_current:.4f}")
    print(f"  선행 NQJ 최고 R²: {r2_lag_best:.4f}")
    print(f"  미래 NQJ 최고 R²: {r2_lead_best:.4f}")

    # 모든 R²이 음수인 경우 단일 피처로 예측 불가 판정
    all_negative = all(v["r2"] < 0 for v in results.values())
    if all_negative:
        verdict = "SINGLE_FEATURE_INSUFFICIENT"
        print("\n  💡 단일 NQJ 피처만으로는 NOx 예측 불가 (R² < 0)")
        print("     → NQJ는 다변수 모델에서 다른 피처와 결합해야 유효")
        print("     → 피처 중요도(SHAP) 기반 분석 권장")
    elif r2_lead_best > r2_lag_best * 1.2 and r2_lag_best > 0:
        verdict = "REVERSE_CAUSALITY_HIGH"
        print("\n  ⚠️  역인과 강함!")
        print("     → NOx 상승 후 제어기가 N2를 증가시킴")
        print("     → 모델에 lag 기반 N2만 포함 권장")
    elif r2_lead_best > r2_lag_best * 1.05 and r2_lag_best > 0:
        verdict = "REVERSE_CAUSALITY_MEDIUM"
        print("\n  ⚠️  역인과 중간 수준")
        print("     → lag N2 중심으로 피처 설계")
    elif r2_lag_best >= r2_current * 0.95:
        verdict = "CAUSAL"
        print("\n  ✅ N2는 원인 효과 (lag ≥ 동시간)")
        print("     → lag + 동시간 N2 모두 사용 가능")
    else:
        verdict = "UNCLEAR"
        print("\n  🤔 인과 방향 불명확 — 추가 검토 필요")

    # ── 6. 시각화 ───────────────────────────────────────────────────
    _plot(results, df)

    return {
        "verdict": verdict,
        "r2_current": r2_current,
        "r2_lag_best": r2_lag_best,
        "r2_lead_best": r2_lead_best,
    }


def _plot(results: dict, df: pd.DataFrame) -> None:
    feat_order = sorted(results.keys())
    r2_vals  = [results[k]["r2"] for k in feat_order]
    bar_colors = [
        "red"       if "lead" in k
        else "steelblue" if "lag" in k
        else "gray"
        for k in feat_order
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. 피처별 R²
    axes[0].barh(range(len(feat_order)), r2_vals, color=bar_colors)
    axes[0].set_yticks(range(len(feat_order)))
    axes[0].set_yticklabels(feat_order, fontsize=8)
    axes[0].set_xlabel("R² Score")
    axes[0].set_title("가설3: 피처별 NOx 예측력\n(파랑=lag/선행, 빨강=lead/미래)")
    legend_patches = [
        plt.matplotlib.patches.Patch(color="steelblue", label="lag (선행-원인)"),
        plt.matplotlib.patches.Patch(color="red",       label="lead (미래-역인과)"),
        plt.matplotlib.patches.Patch(color="gray",      label="동시간"),
    ]
    axes[0].legend(handles=legend_patches, fontsize=7)

    # 2. 상관 계수
    corr_vals = [results[k]["corr"] for k in feat_order]
    axes[1].barh(range(len(feat_order)), corr_vals, color=bar_colors)
    axes[1].set_yticks(range(len(feat_order)))
    axes[1].set_yticklabels(feat_order, fontsize=8)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Pearson 상관계수")
    axes[1].set_title("피처별 NOx 상관계수")

    # 3. NOx vs 동시간 NQJ
    sample_n = min(5000, len(df))
    s = df.sample(sample_n, random_state=42)
    nqj_col = NQJ_COL
    nox_col  = TARGET
    if nqj_col in df.columns:
        corr_v = results.get(nqj_col, {}).get("corr", np.nan)
        axes[2].scatter(s[nqj_col], s[nox_col], alpha=0.3, s=5, color="gray")
        axes[2].set_xlabel("N2 주입 유량 (NQJ, 동시간)")
        axes[2].set_ylabel("NOx [ppm]")
        axes[2].set_title(f"NOx vs N2 동시간  (r={corr_v:.3f})")

    plt.tight_layout()
    save_path = REPORT_DIR / "h3_n2_causality.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📈 그래프 저장: {save_path}")


if __name__ == "__main__":
    run()
