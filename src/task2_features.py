"""Task 2 (Unit 4): HOG, SIFT keypoints/descriptors, matching (+ rotation / resize robustness)."""
import os, cv2, numpy as np, pandas as pd
from skimage.feature import hog
from skimage import exposure
from common import *
import matplotlib.pyplot as plt

O = os.path.join(OUT, "task2"); os.makedirs(O, exist_ok=True)
a = load("durga_1.jpg", 800)           # portrait view
b = load("durga_2.jpg", 800)           # same scene, different framing/angle (landscape)
ga, gb = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (a, b)]

# 1. HOG
g = cv2.resize(ga, (256, 384))
feat, hog_img = hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
hog_img = exposure.rescale_intensity(hog_img, in_range=(0, np.percentile(hog_img, 99)))
fig, ax = plt.subplots(1, 2, figsize=(8, 6)); ax[0].imshow(g, cmap="gray"); ax[0].set_title("Input (256x384)")
ax[1].imshow(hog_img, cmap="gray"); ax[1].set_title(f"HOG ({len(feat)}-D vector)")
[x.axis("off") for x in ax]; plt.tight_layout(); plt.savefig(os.path.join(O, "hog.png"), dpi=110); plt.close()
print("HOG vector length:", len(feat))

# 2. SIFT
sift = cv2.SIFT_create(nfeatures=4000)
def kp(g): return sift.detectAndCompute(g, None)
ka, da = kp(ga); kb, db = kp(gb)
print("SIFT keypoints:", len(ka), len(kb), "descriptor shape", da.shape)
cv2.imwrite(os.path.join(O, "sift_keypoints_img1.png"),
            cv2.drawKeypoints(a, ka[:1500], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
cv2.imwrite(os.path.join(O, "sift_keypoints_img2.png"),
            cv2.drawKeypoints(b, kb[:1500], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))

# 3/4. matching with Lowe ratio test (0.75) + RANSAC homography
def match(imgA, kA, dA, imgB, name):
    gB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY); kB, dB = kp(gB)
    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dA, dB, k=2)
    good = [m for m, n in (r for r in raw if len(r) == 2) if m.distance < 0.75 * n.distance]
    inl = 0; mask = None
    if len(good) >= 4:
        src = np.float32([kA[m.queryIdx].pt for m in good]); dst = np.float32([kB[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        inl = int(mask.sum()) if mask is not None else 0
    vis = cv2.drawMatches(imgA, kA, imgB, kB, good[:80], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(O, f"match_{name}.png"), vis)
    return [name, len(kB), len(good), inl, round(100 * inl / max(len(good), 1), 1)]

def rotate(im, ang):
    h, w = im.shape[:2]; M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1)
    c, s = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(im, M, (nw, nh))

rows = [match(a, ka, da, b, "img1_vs_img2_original"),
        match(a, ka, da, cv2.resize(b, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), "img1_vs_img2_resized_50pct"),
        match(a, ka, da, rotate(b, 45), "img1_vs_img2_rotated_45deg"),
        match(a, ka, da, rotate(b, 90), "img1_vs_img2_rotated_90deg")]
df = pd.DataFrame(rows, columns=["Experiment", "Keypoints in img2", "Good matches (ratio 0.75)", "RANSAC inliers", "Inlier %"])
df.to_csv(os.path.join(O, "sift_matching_table.csv"), index=False); print(df.to_string(index=False))
