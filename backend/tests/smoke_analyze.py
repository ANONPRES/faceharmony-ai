"""Smoke-test analyze endpoint and print pose / thirds / hairline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import urllib.request


def analyze(path: Path) -> dict:
    """POST an image to /analyze and return JSON."""
    boundary = "----FaceHarmonyBoundary"
    data = path.read_bytes()
    ctype = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/analyze",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.load(resp)


def main() -> None:
    """Run analysis for each CLI path (or default sample)."""
    paths = [Path(p) for p in sys.argv[1:]] or [Path(__file__).parent / "sample_face.jpg"]
    for path in paths:
        payload = analyze(path)
        thirds = payload["overlay"]["thirds_y"]
        print("===", path.name)
        print(
            {
                "overall": payload["overall"],
                "frontal": payload["frontal_score"],
                "profile": payload["profile_score"],
                "pose": payload["pose"],
                "roll": payload.get("roll_deg"),
                "symmetry": payload["symmetry"],
                "thirds": payload["thirds"],
                "eyes": payload["eyes"],
                "jaw": payload["jaw"],
            }
        )
        print("thirds_y", [round(y, 4) for y in thirds])
        print("hairline_vs_brow", round(thirds[1] - thirds[0], 4))


if __name__ == "__main__":
    main()
