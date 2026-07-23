import cv2
from app.services.landmark_detector import FaceLandmarkDetector
from app.services.metrics_calculator import MetricsCalculator

d = FaceLandmarkDetector()
img = cv2.imread("tests/model_high.png")
h, w = img.shape[:2]
r = MetricsCalculator(d.detect(img), w, h, img).calculate_all()
ms = {m["id"]: m for m in r["measurements"]}
print("pose", r["pose"], "count", len(ms))
assert "gonial_angle" not in ms, "gonial should be skipped on frontal"
assert "ear_protrusion" not in ms, "ear should be skipped on frontal"
assert "chin_projection" not in ms, "chin projection should be skipped on frontal"
keys = [
    "face_wh_cheek",
    "cheek_position",
    "fifth_right_outer",
    "golden_face",
    "neck_width",
    "iaa_jfa_diff",
]
for k in keys:
    if k not in ms:
        print("MISSING", k)
        continue
    m = ms[k]
    print(f"{k}: {m['display']} -> {m['score_10']}/10 ideal {m['ideal_min']}-{m['ideal_max']}")
