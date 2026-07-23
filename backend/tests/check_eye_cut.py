"""Quick API check for eye_cut field."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path


def main() -> None:
    """POST barrett.png and print eye_cut fields."""
    path = Path(__file__).resolve().parent / "barrett.png"
    boundary = "----FaceHarmonyBoundary"
    data = path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="barrett.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8001/analyze",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        payload = json.load(resp)
    print("eye_cut", payload.get("eye_cut"))
    print("cheekbones", payload.get("cheekbones"))
    print("metrics_eye_cut", payload.get("metrics", {}).get("eye_cut"))
    print("metric_keys", sorted(payload.get("metrics", {}).keys()))


if __name__ == "__main__":
    main()
