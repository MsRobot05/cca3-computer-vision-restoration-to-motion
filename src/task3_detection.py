"""Task 3 (Unit 5): Haar cascade (Viola-Jones) vs YOLOv8 on video frames and photos."""
import os, cv2, numpy as np, pandas as pd
from ultralytics import YOLO
from common import *

O = os.path.join(OUT, "task3"); os.makedirs(O, exist_ok=True)
frames, fps = read_frames()
picks = {f"video_t{i / fps:.1f}s": frames[i] for i in sample_times(len(frames), fps)}
for n in ("durga_1", "durga_2"): picks[n] = load(n + ".jpg", 800)

def load_cascade(fname):
    """Load a Haar cascade; print the real reason if it fails instead of crashing."""
    cands = [os.path.join(DATA, "haarcascades", fname)]
    if hasattr(cv2, "data"): cands.append(os.path.join(cv2.data.haarcascades, fname))
    for p in cands:
        try:
            c = cv2.CascadeClassifier(p)
            if not c.empty(): return c
            print(f"[Haar] could not load {p} (file missing or unreadable)")
        except Exception as e:
            print(f"[Haar] error loading {p}: {type(e).__name__}: {e}")
    print(f"[Haar] WARNING: {fname} unavailable - Haar results will be 0 for this detector")
    return None

haar_face = load_cascade("haarcascade_frontalface_default.xml")
haar_body = load_cascade("haarcascade_fullbody.xml")
def haar_detect(c, g, **kw):
    return c.detectMultiScale(g, **kw) if c is not None else []
model = YOLO(os.path.join(ROOT, "yolov8n.pt"))

rows, vis_h, vis_y, names = [], [], [], []
for name, img in picks.items():
    g = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    faces = haar_detect(haar_face, g, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    bodies = haar_detect(haar_body, g, scaleFactor=1.1, minNeighbors=3, minSize=(60, 120))
    h = img.copy()
    for (x, y, w, hh) in faces: cv2.rectangle(h, (x, y), (x + w, y + hh), (0, 255, 0), 3); cv2.putText(h, "face", (x, y - 5), 0, 0.8, (0, 255, 0), 2)
    for (x, y, w, hh) in bodies: cv2.rectangle(h, (x, y), (x + w, y + hh), (255, 0, 0), 3); cv2.putText(h, "body", (x, y - 5), 0, 0.8, (255, 0, 0), 2)
    r = model(img, conf=0.25, verbose=False)[0]
    y_img = r.plot()
    dets = [(r.names[int(c)], float(s)) for c, s in zip(r.boxes.cls, r.boxes.conf)]
    summary = ", ".join(f"{n} {s:.2f}" for n, s in dets) or "none"
    rows.append([name, len(faces), len(bodies), len(dets), summary])
    cv2.imwrite(os.path.join(O, f"haar_{name}.png"), h); cv2.imwrite(os.path.join(O, f"yolo_{name}.png"), y_img)
    vis_h.append(h); vis_y.append(y_img); names.append(name)

df = pd.DataFrame(rows, columns=["Image / frame", "Haar faces", "Haar bodies", "YOLO detections", "YOLO classes + confidence"])
df.to_csv(os.path.join(O, "detection_comparison.csv"), index=False); print(df.to_string(index=False))
for i in range(0, len(names), 1):
    pass
ims, ts = [], []
for n, h, y in zip(names, vis_h, vis_y): ims += [h, y]; ts += [f"{n} - Haar", f"{n} - YOLOv8n"]
grid(ims, ts, os.path.join(OUT, "task3", "detection_grid.png"), cols=4, size=4.2)