from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


# Your fixed temperature digits ROI (excluding °C)
TEMP_DIGITS_ROI = {"x": 1857, "y": 0, "w": 101, "h": 32}


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: int
    h: int


def crop(im: np.ndarray, b: Box) -> np.ndarray:
    x1 = int(round(b.x))
    y1 = int(round(b.y))
    x2 = x1 + int(b.w)
    y2 = y1 + int(b.h)
    x1 = max(0, min(x1, im.shape[1] - 1))
    y1 = max(0, min(y1, im.shape[0] - 1))
    x2 = max(x1 + 1, min(x2, im.shape[1]))
    y2 = max(y1 + 1, min(y2, im.shape[0]))
    return im[y1:y2, x1:x2]


def binarize(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # tiny noise cleanup
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return bw


def load_positions(path: Path) -> Dict[str, Box]:
    d = json.loads(path.read_text(encoding="utf-8"))
    pos = d.get("positions", {})
    return {k: Box(**v) for k, v in pos.items()}


def expected_for_pos(i: int) -> str:
    # 1..19 for YYYY-MM-DD HH:MM:SS
    if i in (5, 8):
        return "-"
    if i == 11:
        return "space"
    if i in (14, 17):
        return ":"
    return "digit"


def label_dir(label: str) -> str:
    return {":": "colon", "-": "dash", "space": "space"}.get(label, label)


def load_templates(root: Path) -> Dict[str, np.ndarray]:
    """
    Load one template per label from glyph_crops/<label_dir>/.
    Template is stored as binarized 0/255 uint8.
    """
    out: Dict[str, np.ndarray] = {}
    for label in [str(i) for i in range(10)] + [":", "-", "space"]:
        d = root / label_dir(label)
        if not d.is_dir():
            continue
        files = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not files:
            continue
        im = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        out[label] = binarize(im)
    return out


def template_score(patch_bw: np.ndarray, tmpl_bw: np.ndarray) -> float:
    # Ensure same size
    if patch_bw.shape != tmpl_bw.shape:
        patch_bw = cv2.resize(patch_bw, (tmpl_bw.shape[1], tmpl_bw.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Lower is better: normalized pixel disagreement
    return float(np.mean(patch_bw != tmpl_bw))


def match_one(patch_bgr: np.ndarray, templates: Dict[str, np.ndarray], allowed: List[str]) -> Tuple[str, float]:
    bw = binarize(patch_bgr)
    best_label = "?"
    best_score = 1e9
    for lab in allowed:
        tmpl = templates.get(lab)
        if tmpl is None:
            continue
        s = template_score(bw, tmpl)
        if s < best_score:
            best_score = s
            best_label = lab
    return best_label, best_score


def parse_timestamp(chars: List[str]) -> datetime:
    s = "".join(chars)
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def extract_temp_digits(im_bgr: np.ndarray, templates: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """
    Extract temperature as string like -9 or 11 from fixed ROI using connected components + template matching.
    Returns (temp_str, worst_score).
    """
    roi = Box(**TEMP_DIGITS_ROI)
    cut = crop(im_bgr, roi)
    bw = binarize(cut)
    # connected components on white pixels
    n, labels, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i].tolist()
        # Keep both digits and thin minus-like strokes.
        if area < 6:
            continue
        is_digit_like = (h >= 10 and w >= 3)
        is_minus_like = (h >= 2 and w >= 8 and w >= (2 * h))
        if not (is_digit_like or is_minus_like):
            continue
        comps.append((x, y, w, h, area))
    if not comps:
        return "", 1.0
    # sort left->right
    comps.sort(key=lambda t: t[0])

    parts: List[Tuple[str, float]] = []
    for x, y, w, h, _ in comps:
        glyph = cut[y : y + h, x : x + w]
        # Very flat glyphs should only match minus to avoid digit confusion.
        allowed = ["-"] if (h <= 7 and w >= 2 * h) else (["-"] + [str(i) for i in range(10)])
        lab, sc = match_one(glyph, templates, allowed=allowed)
        # Keep only plausible glyphs
        if lab == "?":
            continue
        parts.append((lab, sc))

    if not parts:
        return "", 1.0

    # Build string, collapse multiple '-' if any
    s = "".join([p[0] for p in parts])
    s = s.replace("--", "-")
    # Keep only leading '-' and digits
    s2 = ""
    for ch in s:
        if ch == "-" and not s2:
            s2 += ch
        elif ch.isdigit():
            s2 += ch
    # Reduce to -?\d{1,2} taking rightmost 2 digits (overlay is right aligned)
    sign = "-" if s2.startswith("-") else ""
    digits = "".join([c for c in s2 if c.isdigit()])
    digits = digits[-2:] if len(digits) > 2 else digits
    temp = sign + digits
    worst = max([p[1] for p in parts]) if parts else 1.0
    return temp, worst


def main() -> None:
    ap = argparse.ArgumentParser(description="Template-match Reconyx timestamp + temperature, rename, and CSV.")
    ap.add_argument("--input", default="100RECNX", help="Input images folder (default: 100RECNX)")
    ap.add_argument("--positions", default="positions.json", help="positions.json (default: positions.json)")
    ap.add_argument("--glyphs", default="glyph_crops", help="Glyph template folder (default: glyph_crops)")
    ap.add_argument("--csv", default="outputs.csv", help="Output CSV (default: outputs.csv)")
    ap.add_argument("--rename", action="store_true", help="Actually rename files")
    ap.add_argument("--minutes-seconds-zero", action="store_true", help="Force MM and SS to 00 in rename")
    ap.add_argument("--max-images", type=int, default=None, help="Process at most N images (debug)")
    args = ap.parse_args()

    folder = Path(args.input)
    images = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.max_images is not None:
        images = images[: max(0, args.max_images)]
    if not images:
        raise SystemExit(f"No images found in: {folder}")

    pos = load_positions(Path(args.positions))
    templates = load_templates(Path(args.glyphs))
    missing = [lab for lab in [str(i) for i in range(10)] + [":", "-", "space"] if lab not in templates]
    if missing:
        raise SystemExit(f"Missing templates for: {missing}. Ensure glyph_crops has those classes.")

    out_csv = Path(args.csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "original_file",
                "new_file",
                "timestamp",
                "timestamp_normalized",
                "temp_c",
                "ts_worst_score",
                "temp_worst_score",
                "status",
                "error",
            ]
        )

        for p in images:
            try:
                im = cv2.imread(str(p))
                if im is None:
                    raise ValueError("OpenCV cannot read")

                chars: List[str] = []
                worst = 0.0
                for i in range(1, 20):
                    pid = f"pos{i:02d}"
                    box = pos[pid]
                    exp = expected_for_pos(i)
                    if exp == "digit":
                        allowed = [str(d) for d in range(10)]
                    elif exp == "-":
                        allowed = ["-"]
                    elif exp == ":":
                        allowed = [":"]
                    else:
                        allowed = ["space"]
                    lab, sc = match_one(crop(im, box), templates, allowed)
                    worst = max(worst, sc)
                    if lab == "space":
                        chars.append(" ")
                    else:
                        chars.append(lab)

                ts = parse_timestamp(chars)
                ts_norm = ts
                if args.minutes_seconds_zero:
                    ts_norm = ts.replace(minute=0, second=0)

                new_name = f"R{ts_norm.strftime('%y%m%d%H%M%S')}.JPG"
                temp_str, temp_score = extract_temp_digits(im, templates)
                temp_out = temp_str

                if args.rename:
                    target = p.with_name(new_name)
                    if target.exists():
                        # avoid collision
                        target = p.with_name(f"{target.stem}_DUP{p.stem}{target.suffix}")
                    p.rename(target)
                    new_written = target.name
                else:
                    new_written = new_name

                w.writerow(
                    [
                        p.name,
                        new_written,
                        ts.strftime("%Y-%m-%d %H:%M:%S"),
                        ts_norm.strftime("%Y-%m-%d %H:%M:%S"),
                        temp_out,
                        f"{worst:.4f}",
                        f"{temp_score:.4f}",
                        "ok",
                        "",
                    ]
                )
            except Exception as exc:
                w.writerow([p.name, "", "", "", "", "", "", "error", str(exc)])

    print(f"Done. Wrote {out_csv}")
    if not args.rename:
        print("Rename disabled (pass --rename to rename files).")


if __name__ == "__main__":
    main()

