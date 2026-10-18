"""Pull evenly spaced keyframes out of a CCTV clip with ffmpeg.

    python data/extract_frames.py --video clips/Robbery048_x264.mp4 \
        --out data/scenes/robbery048/frames --every 2 --max 8

Optionally cut a time window first (UCA gives start/end times per event):
    --start 12.5 --end 30
"""
import argparse
import glob
import os
import shutil
import subprocess
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=float, default=2.0, help="seconds between frames")
    ap.add_argument("--max", type=int, default=8, help="keep at most this many frames")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found. Install it: sudo apt install ffmpeg  (or conda install ffmpeg)")
    os.makedirs(args.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["ffmpeg", "-loglevel", "error", "-y"]
        if args.start is not None:
            cmd += ["-ss", str(args.start)]
        if args.end is not None:
            cmd += ["-to", str(args.end)]
        cmd += ["-i", args.video, "-vf", f"fps=1/{args.every}", "-q:v", "2",
                os.path.join(tmp, "f_%04d.jpg")]
        subprocess.run(cmd, check=True)
        frames = sorted(glob.glob(os.path.join(tmp, "*.jpg")))
        if not frames:
            raise SystemExit("No frames extracted. Check the video path / time window.")
        step = max(1, len(frames) // args.max)
        keep = frames[::step][: args.max]
        for i, f in enumerate(keep, 1):
            shutil.copy(f, os.path.join(args.out, f"frame_{i:02d}.jpg"))
    print(f"Saved {len(keep)} frames to {args.out}")


if __name__ == "__main__":
    main()
