"""Task 1 (Unit 3): image restoration + morphology + edges + watershed."""
import os, cv2, numpy as np, pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from scipy import ndimage as ndi
from common import *

O = os.path.join(OUT, "task1"); os.makedirs(O, exist_ok=True)
rng = np.random.default_rng(42)

# ---------- A. Image restoration (photo: Durga idol, colour -> evaluated in greyscale and colour)
clean = load("durga_1.jpg", 800)
def gauss_noise(im, sigma=25):
    return np.clip(im.astype(np.float32) + rng.normal(0, sigma, im.shape), 0, 255).astype(np.uint8)
def salt_pepper(im, amount=0.05):
    out = im.copy(); m = rng.random(im.shape[:2])
    out[m < amount / 2] = 0; out[m > 1 - amount / 2] = 255
    return out

noisy = {"Gaussian noise (sigma=25)": gauss_noise(clean),
         "Salt & pepper (5%)": salt_pepper(clean)}
filters = {"Median 3x3": lambda x: cv2.medianBlur(x, 3),
           "Median 5x5": lambda x: cv2.medianBlur(x, 5),
           "Gaussian 5x5 s=1.2": lambda x: cv2.GaussianBlur(x, (5, 5), 1.2)}

rows, imgs, titles = [], [clean], ["Original (clean)"]
for nname, n in noisy.items():
    rows.append([nname, "none (noisy input)", psnr(clean, n), ssim(clean, n, channel_axis=2)])
    imgs.append(n); titles.append(nname)
    for fname, f in filters.items():
        r = f(n); imgs.append(r); titles.append(f"{nname}\n-> {fname}")
        rows.append([nname, fname, psnr(clean, r), ssim(clean, r, channel_axis=2)])
        cv2.imwrite(os.path.join(O, f"restored_{nname[:3]}_{fname.replace(' ','_')}.png"), r)
    cv2.imwrite(os.path.join(O, f"noisy_{nname[:3]}.png"), n)
df = pd.DataFrame(rows, columns=["Noise", "Filter", "PSNR (dB)", "SSIM"]).round(3)
df.to_csv(os.path.join(O, "restoration_metrics.csv"), index=False)
print(df.to_string(index=False))
# 3x? grid: clean, then per noise: noisy + 3 restored
order = [0, 1, 2, 3, 4, 5, 6, 7, 8]
grid(imgs, titles, os.path.join(O, "restoration_grid.png"), cols=4, size=4)

# ---------- B. Morphology / threshold / canny / watershed (photo: coins, touching objects)
coins_full = load("coins.jpg")
# Crop a region of interest around the coins (table lighting is uneven: bright wood streaks at the top)
coins = coins_full[500:900, 20:570].copy()
gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
if binary.mean() > 127: binary = 255 - binary          # make objects white
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
ero = cv2.erode(binary, k); dil = cv2.dilate(binary, k)
opn = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_big); cls = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
edges = cv2.Canny(blur, 40, 120)
edges_d = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY), (5, 5), 0), 80, 200)

# watershed: distance transform -> markers -> cv2.watershed
clean_bin = cv2.morphologyEx(opn, cv2.MORPH_CLOSE, k)  # opening removes streaks, closing joins gaps
clean_bin = (ndi.binary_fill_holes(clean_bin > 0) * 255).astype(np.uint8)  # fill dark coin interiors
dist = cv2.distanceTransform(clean_bin, cv2.DIST_L2, 5)
# markers = local maxima of the distance map (one per coin, even if the coins overlap)
from skimage.feature import peak_local_max
dist_s = cv2.GaussianBlur(dist, (0, 0), 5)
pk = peak_local_max(dist_s, min_distance=60, threshold_rel=0.2, labels=(clean_bin > 0).astype(int))
fg = np.zeros_like(clean_bin)
for y, x in pk: cv2.circle(fg, (int(x), int(y)), 8, 255, -1)
bg = cv2.dilate(clean_bin, k, iterations=2)
unknown = cv2.subtract(bg, fg)
n, markers = cv2.connectedComponents(fg)
markers = markers + 1; markers[unknown == 255] = 0
# classic watershed on the inverted distance map, restricted to the foreground mask
from skimage.segmentation import watershed as sk_watershed
markers = sk_watershed(-dist_s, n_lab := cv2.connectedComponents(fg)[1], mask=clean_bin > 0)
n_objects = markers.max()
ws = coins.copy()
bnd = cv2.morphologyEx(markers.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
ws[bnd] = (0, 0, 255)
colors = np.random.default_rng(1).integers(60, 255, (n_objects + 1, 3)); colors[0] = 0
overlay = cv2.addWeighted(coins, 0.6, colors[markers].astype(np.uint8), 0.4, 0)
overlay[bnd] = (0, 0, 255); ws = overlay
print("Watershed objects found:", n_objects)
cc = ndi.label(clean_bin)[1]
print("Connected components WITHOUT watershed:", cc)

for nm, im in dict(binary=binary, erosion=ero, dilation=dil, opening=opn, closing=cls,
                   canny_coins=edges, canny_photo=edges_d, distance=cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                   watershed=ws).items():
    cv2.imwrite(os.path.join(O, f"{nm}.png"), im)
grid([coins, binary, ero, dil, opn, cls], ["Original coins", "Otsu binary mask", "Erosion 9x9", "Dilation 9x9", "Opening (25x25)", "Closing 9x9"],
     os.path.join(O, "morphology_grid.png"), cols=3, size=3.5)
grid([clean, edges_d, edges, cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), ws],
     ["Photo", "Canny (photo)", "Canny (coins)", "Distance transform", f"Watershed ({n_objects} objects; red = boundaries)"],
     os.path.join(O, "edges_watershed_grid.png"), cols=3, size=3.5)
pd.DataFrame([["Connected components (no watershed)", cc], ["Watershed regions", n_objects]],
             columns=["Method", "Objects"]).to_csv(os.path.join(O, "segmentation_counts.csv"), index=False)
