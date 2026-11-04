"""Task 4 (Unit 5): frame differencing, GMM (MOG2) background subtraction, Lucas-Kanade optical flow."""
import os, cv2, numpy as np, pandas as pd
from common import *

O = os.path.join(OUT, "task4"); os.makedirs(O, exist_ok=True)
frames, fps = read_frames()
frames = [cv2.resize(f, (360, 640)) for f in frames]
gray = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0) for f in frames]
K = np.ones((3, 3), np.uint8)
mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32, detectShadows=True)
lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
fp = dict(maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7)

sel = set(sample_times(len(frames), fps, (0.15, 0.3, 0.45, 0.65)))
stats, rows, out_vid = [], [], []
prev_pts = None
writer = cv2.VideoWriter(os.path.join(O, "motion_demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (360 * 4, 640))
for i in range(1, len(frames)):
    # 1. frame differencing
    d = cv2.absdiff(gray[i], gray[i - 1]); _, fd = cv2.threshold(d, 20, 255, cv2.THRESH_BINARY)
    fd = cv2.morphologyEx(fd, cv2.MORPH_OPEN, K)
    # 2. GMM background subtraction (shadows=127 removed)
    m = mog.apply(frames[i]); _, gm = cv2.threshold(m, 200, 255, cv2.THRESH_BINARY)
    gm = cv2.morphologyEx(gm, cv2.MORPH_OPEN, K)
    # 3. Lucas-Kanade sparse optical flow
    p0 = cv2.goodFeaturesToTrack(gray[i - 1], mask=None, **fp)
    vis = frames[i].copy(); mags = []
    if p0 is not None:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(gray[i - 1], gray[i], p0, None, **lk)
        for a, b in zip(p0[st == 1], p1[st == 1]):
            mag = np.linalg.norm(b - a); mags.append(mag)
            cv2.circle(vis, tuple(map(int, a)), 2, (0, 255, 255), -1)
            if mag > 0.5: cv2.arrowedLine(vis, tuple(map(int, a)), tuple(map(int, a + 4 * (b - a))), (0, 0, 255), 2, tipLength=0.35)  # vector drawn 4x for visibility
    mags = np.array(mags) if mags else np.zeros(1)
    stats.append([i, 100 * (fd > 0).mean(), 100 * (gm > 0).mean(), mags.mean(), np.percentile(mags, 90)])
    comb = np.hstack([frames[i], cv2.cvtColor(fd, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gm, cv2.COLOR_GRAY2BGR), vis])
    writer.write(comb)
    if i in sel:
        for nm, im in (("frame", frames[i]), ("framediff", fd), ("gmm_mask", gm), ("lk_flow", vis)):
            cv2.imwrite(os.path.join(O, f"{nm}_{i:03d}.png"), im)
        rows.append((i, [frames[i], fd, gm, vis]))
writer.release()
ims, ts = [], []
for i, four in rows:
    ims += four; ts += [f"Frame {i} (t={i/fps:.1f}s)", "Frame differencing", "GMM / MOG2 mask", "Lucas-Kanade vectors"]
grid(ims, ts, os.path.join(O, "motion_grid.png"), cols=4, size=3.2)
df = pd.DataFrame(stats, columns=["frame", "framediff_fg_%", "gmm_fg_%", "LK_mean_px", "LK_p90_px"]).round(2)
df.to_csv(os.path.join(O, "motion_stats_per_frame.csv"), index=False)
summ = df.drop(columns="frame").agg(["mean", "max"]).round(2); summ.to_csv(os.path.join(O, "motion_summary.csv")); print(summ)
