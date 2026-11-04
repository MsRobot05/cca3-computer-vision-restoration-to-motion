"""Shared helpers: paths, image loading, figure saving."""
import os, cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

def load(name, max_side=None):
    img = cv2.imread(os.path.join(DATA, name))
    assert img is not None, f"missing {name} in data/"
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img

def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img

def grid(images, titles, path, cols=3, size=4, cmap="gray"):
    """Save a captioned grid of images."""
    rows = int(np.ceil(len(images) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * size, rows * size * 1.05))
    ax = np.atleast_1d(ax).ravel()
    for a in ax: a.axis("off")
    for a, im, t in zip(ax, images, titles):
        a.imshow(rgb(im), cmap=cmap if im.ndim == 2 else None)
        a.set_title(t, fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)

def read_frames(video="video.mp4"):
    path = os.path.join(DATA, video)
    if not os.path.exists(path):
        raise SystemExit(f"\nERROR: {path} not found.\nCopy your original video into the data/ folder and name it video.mp4, then run again.")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit("ERROR: video.mp4 could not be decoded (corrupt file or missing codec). Try re-exporting it as H.264 mp4.")
    return frames, fps

def sample_times(n_frames, fps, fractions=(0.04, 0.15, 0.3, 0.6, 0.9)):
    """Sample moments as fractions of the video length so any clip length works."""
    return [min(int(f * n_frames), n_frames - 1) for f in fractions]
