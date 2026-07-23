import cv2
from app.services.landmark_detector import FaceLandmarkDetector
from app.services.metrics_calculator import MetricsCalculator

d = FaceLandmarkDetector()
for name in ["tests/model_high.png", "tests/user_self.png"]:
    img = cv2.imread(name)
    h, w = img.shape[:2]
    ms = {
        m["id"]: m
        for m in MetricsCalculator(d.detect(img), w, h, img).calculate_all()["measurements"]
    }
    keys = [
        "iaa_jfa_diff",
        "iaa",
        "jfa",
        "cheekbone_height",
        "jaw_width_bigonial",
        "canthal_tilt",
        "forehead_width",
        "upper_third",
        "mouth_nose_width",
        "lip_ratio",
    ]
    print("===", name, "count", len(ms))
    for k in keys:
        m = ms[k]
        print(
            f"  {k}: {m['display']} -> {m['score_10']}/10 "
            f"(ideal {m['ideal_min']}-{m['ideal_max']})"
        )
