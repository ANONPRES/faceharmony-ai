"""Compare local celeb photos against FaceIQ Labs published Harmony (front)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.faceiq_reference import CELEB_HARMONY, SEAN_FACEIQ  # noqa: E402
from app.services.landmark_detector import FaceLandmarkDetector  # noqa: E402
from app.services.metrics_calculator import MetricsCalculator  # noqa: E402
from app.services.scoring import range_score  # noqa: E402

# Local file → FaceIQ celeb key (best-effort identity)
BATCH: list[tuple[str, str]] = [
    ("sean_opry", "tests/metric_batch/02_sean.jfif"),
    ("corrado_martini", "tests/metric_batch/11_attach_lia.png"),
]


def main() -> None:
    detector = FaceLandmarkDetector()
    print("=== Harmony (front-only) vs FaceIQ ===")
    print(f"{'file':28s} {'our':>5} {'fiqF':>5} {'d':>5}  fiqH")
    for key, rel in BATCH:
        path = ROOT / rel
        if not path.exists():
            print("missing", rel)
            continue
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        result = MetricsCalculator(
            detector.detect(img), w, h, img, gender="male"
        ).calculate_all()
        ref = CELEB_HARMONY[key]
        our = result["harmony"] / 10.0
        print(
            f"{path.name:28s} {our:5.1f} {ref['front']:5.1f} "
            f"{our - ref['front']:+5.1f}  {ref['harmony']:.2f}"
        )
        if key == "sean_opry":
            ours = {m["id"]: m for m in result["measurements"]}
            print("\n  Sean metric spot-check (score curve vs FIQ value):")
            for mid, ref_m in SEAN_FACEIQ.items():
                if mid == "harmony_faceiq":
                    continue
                m = ours.get(mid)
                if not m:
                    continue
                curve = (
                    range_score(
                        float(ref_m["value"]),
                        m["ideal_min"],
                        m["ideal_max"],
                        m.get("soft_margin"),
                    )
                    / 10.0
                )
                print(
                    f"    {mid:20s} ours={m['value']:.2f} fiq={ref_m['value']:.2f} "
                    f"sc={m['score_10']:.1f}/{ref_m['score_10']:.1f} "
                    f"curve@{ref_m['value']:.2f}={curve:.1f}"
                )


if __name__ == "__main__":
    main()
