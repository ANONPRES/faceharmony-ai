"""Compare our Sean O'Pry measurements against FaceIQ Labs published values."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.faceiq_reference import SEAN_FACEIQ  # noqa: E402
from app.services.landmark_detector import FaceLandmarkDetector  # noqa: E402
from app.services.metrics_calculator import MetricsCalculator  # noqa: E402
from app.services.scoring import range_score  # noqa: E402


def main() -> None:
    photo = ROOT / "tests" / "metric_batch" / "02_sean.jfif"
    if not photo.exists():
        raise SystemExit(f"missing {photo}")

    img = cv2.imread(str(photo))
    h, w = img.shape[:2]
    result = MetricsCalculator(
        FaceLandmarkDetector().detect(img), w, h, img, gender="male"
    ).calculate_all()
    ours = {m["id"]: m for m in result["measurements"]}

    print(f"{'metric':22s} {'ours':>8} {'fiq':>8} {'dVal':>7}  {'sc':>5} {'fiqSc':>5} {'dSc':>5}  note")
    print("-" * 90)
    for mid, ref in SEAN_FACEIQ.items():
        if mid.startswith("harmony"):
            continue
        m = ours.get(mid)
        if not m:
            print(f"{mid:22s} {'MISSING':>8}")
            continue
        ov, fv = float(m["value"]), float(ref["value"])
        os_, fs = float(m["score_10"]), float(ref["score_10"])
        # Score if value were exactly FaceIQ's (validates curve)
        curve = range_score(fv, m["ideal_min"], m["ideal_max"], m.get("soft_margin")) / 10.0
        note = "OK" if abs(os_ - fs) <= 1.0 and abs(ov - fv) / max(abs(fv), 1e-3) < 0.08 else ""
        if abs(curve - fs) <= 0.35:
            note = (note + "+curve").strip("+") if note == "OK" else "curveOK"
        elif abs(ov - fv) / max(abs(fv), 1e-3) > 0.08:
            note = "valueΔ (photo/landmarks)"
        print(
            f"{mid:22s} {ov:8.2f} {fv:8.2f} {ov - fv:+7.2f}  "
            f"{os_:5.1f} {fs:5.1f} {os_ - fs:+5.1f}  {note}"
        )

    print()
    h = result["pillars"].get("harmony")
    print(
        f"Our Harmony: {h/10:.1f}/10  overall={result['overall']:.1f}  "
        f"(A/D/F Coming Soon like FaceIQ)"
    )
    print(
        f"FaceIQ: overall {SEAN_FACEIQ['harmony_faceiq']['score_10']:.2f} "
        f"/ front {SEAN_FACEIQ['harmony_front']['score_10']:.1f} "
        f"(compare front without profile)"
    )

    # Category averages (FaceIQ-style groups)
    from collections import defaultdict

    by_cat: dict[str, list[float]] = defaultdict(list)
    for m in result["measurements"]:
        by_cat[m["category"]].append(float(m["score_10"]))
    print("\nCategory averages:")
    for cat, scores in by_cat.items():
        print(f"  {cat:16s} {sum(scores)/len(scores):.1f}/10  (n={len(scores)})")


if __name__ == "__main__":
    main()
