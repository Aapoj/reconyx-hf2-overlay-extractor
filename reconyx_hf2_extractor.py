from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import re
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import cv2
import numpy as np
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


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


def run_extraction(
    folder: Path,
    positions_path: Path,
    glyphs_path: Path,
    out_csv: Path,
    rename: bool,
    minutes_seconds_zero: bool,
    max_images: int | None,
    no_progress: bool,
    progress_cb=None,
    log_cb=None,
    skip_already_renamed: bool = False,
) -> None:
    images = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if max_images is not None:
        images = images[: max(0, max_images)]
    if not images:
        raise SystemExit(f"No images found in: {folder}")

    # Matches our rename scheme: RYYYYMMDDHH0000[pn]TT.JPG
    renamed_re = re.compile(r"^R\d{10}0000[pn]\d{2}\.JPG$", re.IGNORECASE)

    pos = load_positions(positions_path)
    templates = load_templates(glyphs_path)
    missing = [lab for lab in [str(i) for i in range(10)] + [":", "-", "space"] if lab not in templates]
    if missing:
        raise SystemExit(f"Missing templates for: {missing}. Ensure glyph_crops has those classes.")

    total = len(images)
    done = 0

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

        it = images
        if not no_progress and tqdm is not None:
            it = tqdm(images, desc="Processing", unit="img")

        for p in it:
            try:
                already_renamed = bool(renamed_re.match(p.name))
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
                if minutes_seconds_zero:
                    ts_norm = ts.replace(minute=0, second=0)

                temp_str, temp_score = extract_temp_digits(im, templates)
                temp_out = temp_str

                # New renaming scheme:
                #   RYYYYMMDDHH0000TT  (positive)
                #   RYYYYMMDDHH0000nTT (negative)
                # TT is always two digits (e.g. 4°C -> 04, -9°C -> n09).
                digits = "".join([c for c in temp_out if c.isdigit()])
                if len(digits) == 0:
                    raise ValueError("Temperature parse failed (no digits)")
                digits = digits[-2:] if len(digits) >= 2 else ("0" + digits)
                # n = negative, p = non-negative (including 0)
                temp_suffix = ("n" if temp_out.strip().startswith("-") else "p") + digits

                base = ts_norm.strftime("%Y%m%d%H") + "0000" + temp_suffix
                new_name = f"R{base}.JPG"

                if rename:
                    if skip_already_renamed and already_renamed:
                        new_written = p.name
                    else:
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
            finally:
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total)
                if log_cb is not None:
                    log_cb(f"Processed {p.name} ({done}/{total})")


def run_gui(script_dir: Path) -> None:
    """
    Batch Tkinter GUI:
    - add multiple (input folder -> output CSV) jobs
    - run jobs in parallel
    - per-job progress + overall progress
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Tkinter GUI is not available: {e}")

    q: queue.Queue = queue.Queue()

    # Defaults based on the package folder
    positions_path = script_dir / "positions.json"
    glyphs_path = script_dir / "glyph_crops"
    if not positions_path.is_file():
        raise SystemExit(f"Missing required file: {positions_path}")
    if not glyphs_path.is_dir():
        raise SystemExit(f"Missing required folder: {glyphs_path}")

    root = tk.Tk()
    root.title("Reconyx HF2 Overlay Extractor (Batch)")
    root.geometry("980x620")

    rename_var = tk.BooleanVar(value=False)
    zero_var = tk.BooleanVar(value=True)

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)

    # Controls row
    top = ttk.Frame(frm)
    top.grid(row=0, column=0, sticky="we")
    frm.grid_columnconfigure(0, weight=1)

    ttk.Checkbutton(top, text="Rename files", variable=rename_var).grid(row=0, column=0, sticky="w")
    ttk.Checkbutton(top, text="Force minutes/seconds to 00", variable=zero_var).grid(row=0, column=1, sticky="w", padx=(12, 0))

    overall = ttk.Progressbar(top, mode="determinate")
    overall.grid(row=0, column=2, sticky="we", padx=(18, 0))
    top.grid_columnconfigure(2, weight=1)

    # Scrollable jobs area
    jobs_outer = ttk.Frame(frm)
    jobs_outer.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
    frm.grid_rowconfigure(1, weight=1)

    canvas = tk.Canvas(jobs_outer, highlightthickness=0)
    scroll = ttk.Scrollbar(jobs_outer, orient="vertical", command=canvas.yview)
    jobs_frame = ttk.Frame(canvas)
    jobs_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.create_window((0, 0), window=jobs_frame, anchor="nw")
    canvas.configure(yscrollcommand=scroll.set)
    canvas.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    ttk.Label(jobs_frame, text="Jobs (each runs in parallel)").grid(row=0, column=0, sticky="w")

    log_box = tk.Text(frm, height=8, wrap="word")
    log_box.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
    frm.grid_rowconfigure(2, weight=0)

    def append_log(s: str) -> None:
        log_box.insert("end", s + "\n")
        log_box.see("end")

    class JobUI:
        def __init__(self, idx: int):
            self.idx = idx
            self.input_var = tk.StringVar(value="")
            self.csv_var = tk.StringVar(value="")
            self.status_var = tk.StringVar(value="idle")
            self.done = 0
            self.total = 0

            r = idx + 1
            row = ttk.Frame(jobs_frame, padding=(0, 8))
            row.grid(row=r, column=0, sticky="we")
            jobs_frame.grid_columnconfigure(0, weight=1)

            ttk.Label(row, text=f"Job {idx+1}").grid(row=0, column=0, sticky="w")
            ttk.Label(row, textvariable=self.status_var).grid(row=0, column=1, sticky="w", padx=(10, 0))

            in_entry = ttk.Entry(row, textvariable=self.input_var, width=70)
            in_entry.grid(row=1, column=0, columnspan=2, sticky="we", pady=(4, 0))
            ttk.Button(row, text="Input folder...", command=self.choose_input).grid(row=1, column=2, padx=(8, 0))

            out_entry = ttk.Entry(row, textvariable=self.csv_var, width=70)
            out_entry.grid(row=2, column=0, columnspan=2, sticky="we", pady=(4, 0))
            ttk.Button(row, text="Output CSV...", command=self.choose_csv).grid(row=2, column=2, padx=(8, 0))

            self.bar = ttk.Progressbar(row, mode="determinate")
            self.bar.grid(row=3, column=0, columnspan=3, sticky="we", pady=(6, 0))

            ttk.Button(row, text="Remove", command=lambda: remove_job(self.idx)).grid(row=0, column=2, sticky="e")
            row.grid_columnconfigure(1, weight=1)

        def choose_input(self):
            p = filedialog.askdirectory(title="Select input images folder")
            if p:
                self.input_var.set(p)
                if not self.csv_var.get().strip():
                    out = Path(p) / "outputs.csv"
                    self.csv_var.set(str(out))

        def choose_csv(self):
            p = filedialog.asksaveasfilename(
                title="Select output CSV file",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            )
            if p:
                self.csv_var.set(p)

    jobs: List[JobUI] = []
    running = {"active": False}
    pool: ThreadPoolExecutor | None = None

    def add_job(default_in: str = "", default_out: str = ""):
        j = JobUI(len(jobs))
        if default_in:
            j.input_var.set(default_in)
        if default_out:
            j.csv_var.set(default_out)
        jobs.append(j)

    def remove_job(idx: int):
        if running["active"]:
            return
        # Rebuild UI list (simplest + robust).
        if idx < 0 or idx >= len(jobs):
            return
        jobs.pop(idx)
        for w in jobs_frame.grid_slaves():
            info = w.grid_info()
            if int(info.get("row", 0)) >= 1:
                w.destroy()
        ttk.Label(jobs_frame, text="Jobs (each runs in parallel)").grid(row=0, column=0, sticky="w")
        old = jobs[:]
        jobs.clear()
        for i, prev in enumerate(old):
            add_job(prev.input_var.get(), prev.csv_var.get())

    def set_overall():
        total = sum(max(0, j.total) for j in jobs)
        done = sum(max(0, j.done) for j in jobs)
        overall["maximum"] = max(1, total)
        overall["value"] = done

    def worker(job_id: int, in_dir: Path, out_csv: Path):
        try:
            # Pre-scan total for overall progress.
            imgs = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            q.put(("job_total", job_id, len(imgs)))

            run_extraction(
                folder=in_dir,
                positions_path=positions_path,
                glyphs_path=glyphs_path,
                out_csv=out_csv,
                rename=bool(rename_var.get()),
                minutes_seconds_zero=bool(zero_var.get()),
                max_images=None,
                no_progress=True,  # GUI has its own progress bars
                progress_cb=lambda d, t: q.put(("job_progress", job_id, d, t)),
                skip_already_renamed=True,
            )
            q.put(("job_done", job_id, str(out_csv)))
        except Exception as exc:
            q.put(("job_error", job_id, str(exc)))

    def start_all():
        nonlocal pool
        if running["active"]:
            return
        # Validate inputs
        specs = []
        for j in jobs:
            in_s = j.input_var.get().strip()
            out_s = j.csv_var.get().strip()
            if not in_s or not out_s:
                messagebox.showerror("Missing fields", "Each job must have an input folder and output CSV.")
                return
            specs.append((Path(in_s), Path(out_s)))

        log_box.delete("1.0", "end")
        for j in jobs:
            j.status_var.set("queued")
            j.done = 0
            j.total = 0
            j.bar["value"] = 0
            j.bar["maximum"] = 1
        set_overall()

        running["active"] = True
        add_btn.config(state="disabled")
        start_btn.config(state="disabled")

        max_workers = max(1, min(len(specs), (os.cpu_count() or 4)))
        pool = ThreadPoolExecutor(max_workers=max_workers)
        append_log(f"Starting {len(specs)} job(s) with up to {max_workers} worker(s).")
        for job_id, (inp, outp) in enumerate(specs):
            jobs[job_id].status_var.set("running")
            pool.submit(worker, job_id, inp, outp)

    def poll():
        nonlocal pool
        try:
            while True:
                item = q.get_nowait()
                kind = item[0]
                if kind == "job_total":
                    _, jid, total = item
                    j = jobs[jid]
                    j.total = int(total)
                    j.bar["maximum"] = max(1, j.total)
                    set_overall()
                elif kind == "job_progress":
                    _, jid, done, total = item
                    j = jobs[jid]
                    j.done = int(done)
                    j.total = int(total)
                    j.bar["maximum"] = max(1, j.total)
                    j.bar["value"] = j.done
                    set_overall()
                elif kind == "job_done":
                    _, jid, out_csv_path = item
                    jobs[jid].status_var.set("done")
                    append_log(f"Job {jid+1} done: {out_csv_path}")
                elif kind == "job_error":
                    _, jid, err = item
                    jobs[jid].status_var.set("error")
                    append_log(f"Job {jid+1} ERROR: {err}")
                    messagebox.showerror(f"Job {jid+1} failed", err)
                # When all jobs are finished (done/error), re-enable controls.
                if running["active"]:
                    states = [j.status_var.get() for j in jobs]
                    if all(s in {"done", "error"} for s in states) and states:
                        running["active"] = False
                        if pool is not None:
                            pool.shutdown(wait=False, cancel_futures=False)
                            pool = None
                        add_btn.config(state="normal")
                        start_btn.config(state="normal")
                        append_log("All jobs finished.")
        except queue.Empty:
            pass
        root.after(100, poll)

    btns = ttk.Frame(frm)
    btns.grid(row=3, column=0, sticky="we", pady=(10, 0))
    btns.grid_columnconfigure(0, weight=1)

    add_btn = ttk.Button(btns, text="Add job", command=lambda: add_job())
    add_btn.grid(row=0, column=0, sticky="w")
    start_btn = ttk.Button(btns, text="Start all", command=start_all)
    start_btn.grid(row=0, column=1, sticky="e")

    # Start with one job pre-filled (nice default)
    add_job(default_in=str(script_dir.parent / "100RECNX"), default_out=str(script_dir / "outputs.csv"))

    root.after(100, poll)
    root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser(description="Template-match Reconyx timestamp + temperature, rename, and CSV.")
    ap.add_argument("--input", default="100RECNX", help="Input images folder (default: 100RECNX)")
    ap.add_argument("--positions", default="positions.json", help="positions.json (default: positions.json)")
    ap.add_argument("--glyphs", default="glyph_crops", help="Glyph template folder (default: glyph_crops)")
    ap.add_argument("--csv", default="outputs.csv", help="Output CSV (default: outputs.csv)")
    ap.add_argument("--rename", action="store_true", help="Actually rename files")
    ap.add_argument("--minutes-seconds-zero", action="store_true", help="Force MM and SS to 00 in rename")
    ap.add_argument("--max-images", type=int, default=None, help="Process at most N images (debug)")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    ap.add_argument("--gui", action="store_true", help="Launch a simple GUI")
    ap.add_argument("--skip-already-renamed", action="store_true", help="When renaming, skip files already matching the RYYYYMMDDHH0000[pn]TT.JPG pattern")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.gui:
        run_gui(script_dir)
        return

    folder = Path(args.input)
    positions_path = Path(args.positions)
    glyphs_path = Path(args.glyphs)
    # Make defaults work even if user runs from another cwd.
    if not positions_path.is_file() and args.positions == "positions.json":
        positions_path = script_dir / "positions.json"
    if not glyphs_path.is_dir() and args.glyphs == "glyph_crops":
        glyphs_path = script_dir / "glyph_crops"

    run_extraction(
        folder=folder,
        positions_path=positions_path,
        glyphs_path=glyphs_path,
        out_csv=Path(args.csv),
        rename=bool(args.rename),
        minutes_seconds_zero=bool(args.minutes_seconds_zero),
        max_images=args.max_images,
        no_progress=bool(args.no_progress),
        skip_already_renamed=bool(args.skip_already_renamed),
    )

    print(f"Done. Wrote {Path(args.csv)}")
    if not args.rename:
        print("Rename disabled (pass --rename to rename files).")


if __name__ == "__main__":
    main()

