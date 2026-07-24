"""FaceIQ Labs ideal bands and Sean O'Pry reference (public celeb wiki).

Values scraped from https://www.faceiqlabs.com/celebs-wiki/sean-opry/*
Angularity / Dimorphism / Features / Overall are still 'Coming Soon' on FaceIQ —
only Harmony ratios are published.
"""

from __future__ import annotations

# FaceIQ ideal bands (min, max) used for scoring. Prefer these over ad-hoc bands.
FACEIQ_IDEALS: dict[str, tuple[float, float]] = {
    # Facial thirds (Sean pages)
    "upper_third": (30.0, 32.0),
    "mid_third": (31.4, 33.4),
    "lower_third": (33.9, 37.0),
    # Eyes
    "eye_spacing": (0.90, 1.05),  # One Eye Apart Test (FaceIQ band)
    "eye_aspect": (3.0, 3.5),
    "canthal_tilt": (4.5, 8.0),  # Lateral Canthal Tilt (wide FaceIQ soft band)
    # Structure
    "jfa": (86.5, 92.5),
    "iaa": (86.5, 92.5),
    "iaa_jfa_diff": (0.0, 2.5),
    "cheekbone_height": (83.0, 100.0),
    # Proportions
    "face_wh_cheek": (1.92, 2.04),  # Face Width to Height (midface FWHR)
    "total_face_wh": (1.31, 1.40),  # Total Facial Width to Height
    "mouth_nose_width": (1.45, 1.55),
    "lip_ratio": (1.30, 1.50),  # Lower / Upper lip
    "nose_width_bridge": (2.0, 2.3),
}

# Soft margins calibrated so Sean's FaceIQ *values* land near FaceIQ *scores*.
FACEIQ_SOFT: dict[str, float] = {
    "eye_spacing": 0.12,  # 1.16 → ~3.0; 1.06 → ~7.8; 0.88 → ~7.3
    "jfa": 12.0,  # 78.9 → ~6.7
    "iaa_jfa_diff": 6.7,  # 10.0 → ~4.2
    "lower_third": 7.0,  # 37.65 → ~9.5
    "upper_third": 2.5,
    "mid_third": 2.5,
    "canthal_tilt": 2.8,
    "face_wh_cheek": 0.10,
    "total_face_wh": 0.05,  # 1.30 → ~7.0 with ideal min 1.31
    "eye_aspect": 0.45,
    "lip_ratio": 0.25,
    "mouth_nose_width": 0.12,
    "cheekbone_height": 12.0,
    "iaa": 5.0,
}

# Celebrity FaceIQ Harmony references (public wiki).
CELEB_HARMONY: dict[str, dict[str, float]] = {
    "sean_opry": {"harmony": 8.10, "front": 7.6, "side": 8.9},
    "corrado_martini": {"harmony": 8.14, "front": 8.7, "side": 7.2},
    "elias_de_poot": {"harmony": 6.91, "front": 6.7, "side": 7.3},
    "martin_garrix": {"harmony": 6.93, "front": 6.5, "side": 7.6},
}

# Sean O'Pry published FaceIQ numbers (for regression / comparison).
SEAN_FACEIQ: dict[str, dict[str, float]] = {
    "jfa": {"value": 78.86, "score_10": 6.7},
    "iaa": {"value": 88.9, "score_10": 10.0},
    "iaa_jfa_diff": {"value": 10.0, "score_10": 4.2},
    "eye_spacing": {"value": 1.16, "score_10": 2.8},  # One Eye Apart
    "eye_aspect": {"value": 3.20, "score_10": 10.0},
    "canthal_tilt": {"value": 7.60, "score_10": 10.0},
    "upper_third": {"value": 30.68, "score_10": 10.0},
    "mid_third": {"value": 31.67, "score_10": 10.0},
    "lower_third": {"value": 37.65, "score_10": 9.5},
    "face_wh_cheek": {"value": 1.99, "score_10": 10.0},
    "total_face_wh": {"value": 1.30, "score_10": 7.0},
    "cheekbone_height": {"value": 97.4, "score_10": 10.0},
    "mouth_nose_width": {"value": 1.5, "score_10": 10.0},
    "lip_ratio": {"value": 1.4, "score_10": 9.6},
    "nose_width_bridge": {"value": 2.2, "score_10": 9.0},
    # Harmony aggregate (FaceIQ publishes this; A/D/F/Overall still Coming Soon)
    "harmony_faceiq": {"value": 8.10, "score_10": 8.10},
}
