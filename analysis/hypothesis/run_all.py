"""
hypothesis/run_all.py
======================
가설 1~5 검증 마스터 러너.

사용법
------
python -m analysis.hypothesis.run_all              # 전체 실행
python -m analysis.hypothesis.run_all --h 1 2 3   # 특정 가설만 실행
python -m analysis.hypothesis.run_all --nrows 100000  # 행 수 제한

결과
----
- 각 가설별 판정 결과 출력
- analysis/reports/hypothesis/ 에 PNG 그래프 저장
- analysis/reports/hypothesis/summary.json 에 판정 결과 저장
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from analysis.hypothesis import h1_o2_leakage
from analysis.hypothesis import h2_output_segments
from analysis.hypothesis import h3_n2_causality
from analysis.hypothesis import h4_temp_lag
from analysis.hypothesis import h5_npr_nonlinear
from analysis.hypothesis.data_loader import TRAIN_FILE, print_section

# REPORT_DIR는 data_loader에 없으므로 재정의
REPORT_DIR_LOCAL = _ROOT / "analysis" / "reports" / "hypothesis"
REPORT_DIR_LOCAL.mkdir(parents=True, exist_ok=True)


HYPOTHESIS_MAP = {
    1: ("O2 준누수 검증",            h1_o2_leakage.run),
    2: ("출력 구간별 NOx 메커니즘",   h2_output_segments.run),
    3: ("N2 주입 역인과 검증",        h3_n2_causality.run),
    4: ("배기가스 온도 지연 검증",    h4_temp_lag.run),
    5: ("NPR 비선형성 검증",          h5_npr_nonlinear.run),
}

# 판정 코드 → 권장 사항
VERDICT_RECOMMENDATIONS = {
    # H1
    "LEAKAGE_HIGH":            "O2 포함/제외 두 모델 모두 유지. 예측 모델은 O2 사용, 해석 모델은 O2 제외.",
    "LEAKAGE_MEDIUM":          "O2 포함/제외 성능 비교 후 선택.",
    "LEAKAGE_LOW":             "O2를 일반 피처로 포함.",
    # H2
    "SEGMENT_EFFECTIVE":       "DWATT hinge 피처 추가 또는 구간별 분리 모델 검토.",
    "SEGMENT_MARGINAL":        "모델 복잡도 고려 후 구간 분리 여부 결정.",
    "GLOBAL_BETTER":           "통합 모델 유지.",
    # H3
    "SINGLE_FEATURE_INSUFFICIENT": "NQJ 단독 예측 불가. SHAP 기반 다변수 기여도 분석 필요.",
    "REVERSE_CAUSALITY_HIGH":  "NQJ는 lag 피처만 사용 (동시간 NQJ 제외).",
    "REVERSE_CAUSALITY_MEDIUM":"NQJ lag 중심으로 피처 설계.",
    "CAUSAL":                  "NQJ lag + 동시간 모두 사용 가능.",
    "UNCLEAR":                 "추가 시계열 분석 후 결정.",
    # H4
    "LAG_EFFECTIVE":           "TTXM_lag 피처 추가 권장.",
    "DIFF_EFFECTIVE":          "TTXM_diff 피처 추가 권장.",
    "CURRENT_SUFFICIENT":      "동시간 TTXM으로 충분.",
    # H5
    "NONLINEAR":               "NPR hinge 또는 다항 피처 추가 권장.",
    "LINEAR":                  "NPR 선형 피처만 사용.",
    "INTERACTION_EFFECTIVE":   "NPR × NQJ / NPR × DWATT 상호작용 피처 추가 권장.",
    "INTERACTION_MARGINAL":    "상호작용 피처는 선택적 추가.",
}


def run_all(
    hypotheses: list[int] | None = None,
    nrows: int | None = 200_000,
    data_path: Path = TRAIN_FILE,
) -> dict:
    """가설 검증 전체 실행.

    Parameters
    ----------
    hypotheses:
        실행할 가설 번호 목록. None이면 전체.
    nrows:
        읽을 최대 행 수.
    data_path:
        CSV 파일 경로.

    Returns
    -------
    dict
        각 가설별 판정 결과.
    """
    if hypotheses is None:
        hypotheses = list(HYPOTHESIS_MAP.keys())

    print_section("NOx 핵심 가설 검증 시작")
    print(f"  데이터: {data_path}")
    print(f"  nrows:  {nrows if nrows else '전체'}")
    print(f"  실행 가설: H{hypotheses}")

    all_results: dict[int, dict] = {}
    timing: dict[int, float] = {}

    for h_num in hypotheses:
        title, run_fn = HYPOTHESIS_MAP[h_num]
        t0 = time.time()
        try:
            result = run_fn(data_path=data_path, nrows=nrows)
            all_results[h_num] = result
        except Exception as exc:
            print(f"\n  [H{h_num}] 오류 발생: {exc}")
            all_results[h_num] = {"error": str(exc)}
        timing[h_num] = round(time.time() - t0, 2)

    # ── 최종 요약 출력 ────────────────────────────────────────────
    print_section("가설 검증 최종 요약")
    print(f"\n  {'#':<5} {'제목':<28} {'주요 판정':<30} {'소요(초)':<10} {'권장 사항'}")
    print("  " + "-" * 110)

    summary_data: dict[str, dict] = {}
    for h_num in hypotheses:
        title = HYPOTHESIS_MAP[h_num][0]
        r = all_results.get(h_num, {})
        elapsed = timing.get(h_num, 0)

        if "error" in r:
            verdict_str = f"오류: {r['error']}"
            rec = "-"
        else:
            # verdict는 str 또는 list[str]
            raw_verdict = r.get("verdict") or r.get("verdicts") or "-"
            if isinstance(raw_verdict, list):
                verdict_str = ", ".join(raw_verdict)
                rec = " / ".join(
                    VERDICT_RECOMMENDATIONS.get(v, "-") for v in raw_verdict
                )
            else:
                verdict_str = str(raw_verdict)
                rec = VERDICT_RECOMMENDATIONS.get(verdict_str, "-")

        print(f"  H{h_num:<4} {title:<28} {verdict_str:<30} {elapsed:<10} {rec}")
        summary_data[f"H{h_num}"] = {
            "title": title,
            "result": {k: (str(v) if not isinstance(v, (int, float, str, list, dict)) else v)
                       for k, v in r.items()},
            "elapsed_sec": elapsed,
            "recommendation": rec,
        }

    # ── 결과 JSON 저장 ────────────────────────────────────────────
    save_path = REPORT_DIR_LOCAL / "summary.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"\n  📄 요약 저장: {save_path}")

    # ── 다음 단계 권고 ────────────────────────────────────────────
    print_section("다음 단계 권고")
    print("""
  1. H1 판정에 따라 생산 모델의 O2 포함 여부 결정
  2. H2 판정에 따라 DWATT hinge 피처 추가 여부 결정
  3. H3 판정에 따라 NQJ를 lag 기반으로 재설계
  4. H4 판정에 따라 TTXM_lag/diff 피처 추가
  5. H5 판정에 따라 NPR 상호작용 피처 추가
  6. 위 결과를 종합해 Feature Engineering 스크립트 작성
     → analysis/hypothesis/feature_engineering.py (추후)
  7. 새 피처셋으로 LightGBM 재학습 및 성능 비교
""")
    return all_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NOx 가설 검증 실행")
    parser.add_argument(
        "--h", nargs="+", type=int, default=None,
        metavar="N", help="실행할 가설 번호 (예: --h 1 2 3)"
    )
    parser.add_argument(
        "--nrows", type=int, default=200_000,
        help="읽을 행 수 (0이면 전체, 기본 200000)"
    )
    parser.add_argument(
        "--data", type=str, default=str(TRAIN_FILE),
        help="CSV 파일 경로"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    nrows_arg = args.nrows if args.nrows > 0 else None
    run_all(
        hypotheses=args.h,
        nrows=nrows_arg,
        data_path=Path(args.data),
    )
