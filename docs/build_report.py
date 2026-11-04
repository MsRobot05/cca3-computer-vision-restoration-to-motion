"""Builds docs/Technical_Report.docx from outputs/ (run after src/run_all.py)."""
import os, pandas as pd
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); O = os.path.join(R, "outputs")
GIT_URL = "<PASTE GIT REPOSITORY URL HERE>"; VIDEO_URL = "<PASTE PROJECT EXPLANATION VIDEO LINK HERE>"
NAME = "<YOUR NAME>  |  <ROLL NO / PRN>"
d = Document(); st = d.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
n_fig = [0]
def H(t, l=1): d.add_heading(t, l)
def P(t, bold=False):
    p = d.add_paragraph(); r = p.add_run(t); r.bold = bold; return p
def B(items):
    for i in items: d.add_paragraph(i, style="List Bullet")
def FIG(path, cap, w=6.2):
    n_fig[0] += 1; d.add_picture(os.path.join(O, path), width=Inches(w)); d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = P(f"Figure {n_fig[0]}: {cap}"); c.alignment = WD_ALIGN_PARAGRAPH.CENTER; c.runs[0].italic = True
def TAB(csv, cap, index=False):
    df = pd.read_csv(os.path.join(O, csv)); P(cap, True)
    t = d.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns): t.rows[0].cells[i].text = str(c)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row): cells[i].text = f"{v:.2f}" if isinstance(v, float) else str(v)
    d.add_paragraph()

d.add_heading("CCA 3 – Computer Vision Practical Assignment", 0)
P("Image Restoration, Segmentation, Feature Extraction, Object Recognition and Motion Estimation (Units 3, 4, 5)")
P(f"Student: {NAME}"); P("Language / libraries: Python 3, OpenCV, NumPy, Matplotlib, scikit-image, Ultralytics YOLO")
P(f"Git repository: {GIT_URL}", True); P(f"Project explanation video: {VIDEO_URL}", True)

H("1. Introduction and Objectives")
P("This report documents the implementation of four computer-vision tasks on a self-captured photograph set and a self-recorded video: (1) image restoration, morphology, edge detection and watershed segmentation; (2) HOG and SIFT feature extraction and matching; (3) object recognition with a Viola–Jones Haar cascade and a pretrained YOLOv8 network; (4) motion estimation with frame differencing, Gaussian-Mixture background subtraction and Lucas–Kanade optical flow. Each section gives the input, the technique, the code location, the outputs and a comparison.")

H("2. Input Data")
B(["Photograph A – durga_1.jpg (1200×1600, portrait): a Durga Puja idol group in a decorated pandal; rich texture, many colours. Used for restoration, HOG, SIFT and detection.",
   "Photograph B – durga_2.jpg (1600×1200, landscape): the same idol from a wider, different viewpoint. Used as the second image for SIFT matching.",
   "Photograph C – coins.jpg (588×1278): three ₹2 coins on a dark wooden table, two of them overlapping. Used for thresholding, morphology and watershed, because touching objects are exactly the case watershed is designed for.",
   "Video – video.mp4 (480×854, ≈29.8 fps, 413 frames, 13.8 s): hand-held street scene with a pedestrian, cars, taxis and motorcycles moving against a static building. Used for detection and motion estimation.",
   "All originals are unedited and stored in data/. Frames are resized to 360×640 for the motion tasks to keep processing fast."])
FIG("../data/durga_1.jpg", "Original photograph A (durga_1.jpg)", 3.0)
FIG("../data/coins.jpg", "Original photograph C (coins.jpg) – used for segmentation", 2.4)

H("3. Task 1 – Image Restoration and Segmentation (Unit 3)")
H("3.1 Image restoration", 2)
P("Technique. Additive white Gaussian noise (σ = 25) and salt-and-pepper noise (5 % of pixels) were synthesised on the clean photograph (fixed random seed 42 for reproducibility). Each noisy image was restored with a 3×3 median filter, a 5×5 median filter and a 5×5 Gaussian filter (σ = 1.2). Quality was measured against the clean image with PSNR (higher is better) and SSIM (closer to 1 is better), computed over the three colour channels.")
P("Code: src/task1_restoration_segmentation.py (section A).")
FIG("task1/restoration_grid.png", "Clean image, noisy images and all restored outputs", 6.5)
TAB("task1/restoration_metrics.csv", "Table 1 – PSNR and SSIM of noisy and restored images")
P("Observation. For Gaussian noise the 5×5 Gaussian filter gave the best result (24.55 dB, SSIM 0.716), a small margin over the 3×3 median (24.18 dB). For salt-and-pepper noise the 3×3 median filter was clearly best (26.72 dB, SSIM 0.893): impulse outliers are removed outright by the median, whereas a Gaussian filter smears them into neighbouring pixels (SSIM only 0.669). Larger kernels (5×5 median) over-smooth the fine ornament detail and lower the scores. Conclusion: match the filter to the noise model – Gaussian filter for Gaussian noise, median filter for impulse noise.")

H("3.2 Thresholding, morphology, edges and watershed", 2)
P("Technique. The coin photograph was cropped to the region of interest (the table is unevenly lit), smoothed with a Gaussian blur and binarised with Otsu’s automatic threshold. Erosion, dilation, opening and closing were applied with elliptical structuring elements (9×9; 25×25 for the opening so that the thin wood-grain streaks are removed). Canny edge detection (thresholds 40/120 for the coins, 80/200 for the idol photo) was applied after Gaussian smoothing. For watershed segmentation, holes in the mask were filled, the Euclidean distance transform was computed, one marker was placed at each distance-map peak (peak_local_max), and a marker-controlled watershed on the inverted distance map split the touching coins.")
FIG("task1/morphology_grid.png", "Original, Otsu binary mask and the four morphological operations", 6.2)
FIG("task1/edges_watershed_grid.png", "Canny edge maps, distance transform and watershed result", 6.2)
FIG("task1/watershed.png", "Final watershed segmentation – three coins separated (red lines = region boundaries)", 3.6)
TAB("task1/segmentation_counts.csv", "Table 2 – Objects found with and without watershed")
P("Observation. Erosion shrinks the objects and removes thin structures, dilation grows them, opening (erosion then dilation) removes small bright noise and the wood streaks, closing fills small holes inside the coins. Connected-component labelling counts only 2 objects because the two overlapping coins form a single blob; watershed correctly returns 3. Canny gives clean closed circles on the coins but is very dense on the highly textured idol photograph, where edge maps become hard to interpret. A first attempt that ran cv2.watershed on the colour gradient produced jagged, texture-driven boundaries, so the distance-map watershed was adopted.")

H("4. Task 2 – Feature Extraction and Matching (Unit 4)")
P("Technique. HOG (9 orientations, 16×16 cells, 2×2 block normalisation) was computed on a 256×384 greyscale version of photograph A and visualised; the feature vector has 12,420 dimensions. SIFT (up to 4000 keypoints) produced 128-D descriptors for photographs A and B. Descriptors were matched with a brute-force L2 matcher and Lowe’s ratio test (0.75); a RANSAC homography (5 px threshold) separates geometrically consistent inliers from false matches. The experiment was repeated after resizing image B to 50 % and rotating it by 45° and 90°.")
P("Code: src/task2_features.py.")
FIG("task2/hog.png", "HOG visualisation of photograph A: strong gradients follow the arch, crown and sari edges", 5.0)
FIG("task2/sift_keypoints_img1.png", "SIFT keypoints on photograph A (size/orientation shown)", 3.6)
FIG("task2/match_img1_vs_img2_original.png", "SIFT matches, original A vs B (80 best shown)", 6.2)
FIG("task2/match_img1_vs_img2_resized_50pct.png", "SIFT matches after resizing B to 50 %", 6.2)
FIG("task2/match_img1_vs_img2_rotated_45deg.png", "SIFT matches after rotating B by 45°", 6.2)
TAB("task2/sift_matching_table.csv", "Table 3 – SIFT matching comparison")
P("Observation. Matching between the two views is reliable: 904 ratio-test matches, 868 (96 %) of them RANSAC inliers. After rotation by 45° or 90° the inlier count and ratio stay essentially unchanged (906 / 859), confirming SIFT’s rotation invariance. Halving the image size reduces the number of keypoints and matches (282 inliers) because fine detail disappears, but the inlier ratio stays above 94 %, showing scale robustness. HOG, in contrast, is not rotation invariant and is better suited to fixed-pose objects (e.g. pedestrian detection).")

H("5. Task 3 – Object Recognition: Viola–Jones vs YOLO (Unit 5)")
P("Technique. Five frames from the video (0.5, 2, 4, 8, 12 s) and both idol photographs were processed with (a) OpenCV Haar cascades (frontal face and full body; histogram-equalised greyscale; scaleFactor 1.1) and (b) pretrained YOLOv8-nano (COCO, confidence ≥ 0.25). Bounding boxes, class labels and confidences were drawn.")
P("Code: src/task3_detection.py.")
FIG("task3/detection_grid.png", "Haar (left of each pair) versus YOLOv8n (right) on video frames and photographs", 6.5)
TAB("task3/detection_comparison.csv", "Table 4 – Detection counts and YOLO confidences")
P("Comparison.", True)
B(["Haar detected no faces or bodies in any of the five video frames: the pedestrian is seen from behind and the faces are tiny, and Haar cascades only handle near-frontal, upright faces. This is a clear miss.",
   "YOLO detected the pedestrian (person 0.80) and consistently found cars, taxis, motorcycles and a truck (confidence 0.25–0.82) – it handles multiple object classes, scales and viewpoints in one pass.",
   "Haar false positives: on durga_1 it reported 4 ‘faces’, of which at least one lies on the plain pink canopy and others on ornaments, while YOLO did not label the idol faces at all (they are statues, not people).",
   "YOLO errors: on durga_2 the whole pandal arch was labelled ‘bed’ (0.27) and the idol group ‘umbrella’ (0.50) – wrong classes caused by objects outside the COCO training distribution. Low-confidence detections (< 0.4) on distant vehicles are often partial or duplicated.",
   "Conclusion: YOLO is far more general and accurate for natural scenes; Haar is fast and light but only suitable for controlled frontal-face settings."])

H("6. Task 4 – Video Motion Estimation (Unit 5)")
P("Technique. Frames were read with OpenCV, resized to 360×640 and Gaussian-blurred. (1) Frame differencing: absolute difference of consecutive greyscale frames, threshold 20, morphological opening. (2) GMM background subtraction: cv2.createBackgroundSubtractorMOG2 (history 200, variance threshold 32, shadows detected and removed by thresholding at 200). (3) Lucas–Kanade sparse optical flow: Shi–Tomasi corners (≤200) tracked with a 21×21 window and 3-level pyramid; flow vectors drawn ×4 for visibility (yellow dots = tracked points, red arrows = motion).")
P("Code: src/task4_motion.py. A side-by-side demo video is written to outputs/task4/motion_demo.mp4.")
FIG("task4/motion_grid.png", "Selected frames with frame-difference mask, GMM foreground mask and Lucas–Kanade vectors", 6.5)
TAB("task4/motion_summary.csv", "Table 5 – Mean and maximum per-frame statistics (foreground %, mean/90th-percentile LK displacement in pixels)")
d.add_paragraph()
P("Comparison table", True)
t = d.add_table(rows=1, cols=4); t.style = "Light Grid Accent 1"
for i, h in enumerate(["Aspect", "Frame differencing", "GMM background subtraction", "Lucas–Kanade optical flow"]): t.rows[0].cells[i].text = h
for row in [("Output", "Binary mask of changed pixels", "Binary foreground mask", "Sparse motion vectors (direction + speed)"),
            ("Moving vehicles", "Detected as fragmented edge blobs; interiors of uniform vehicles missed", "Detected as filled blobs once the background model has learned the scene (e.g. taxi, truck)", "Strong vectors on the vehicles, tiny vectors on the static building"),
            ("Camera shake", "High sensitivity – edges of the whole scene flicker", "Moderate – thin false lines along building edges (frame 268)", "Low – only a small common displacement on the background"),
            ("Slow / stopped objects", "Missed", "Absorbed into background over time", "Tracked while features exist"),
            ("Compute cost", "Lowest", "Low–medium", "Medium"),
            ("Best for", "Quick motion cue with a static camera", "Segmenting moving objects with a static camera", "Estimating how objects move")]:
    c = t.add_row().cells
    for i, v in enumerate(row): c[i].text = v
d.add_paragraph()
P("Observation. Because the video was recorded by hand, the global camera shake produces false motion at strong edges (building outlines) for both frame differencing and GMM. GMM yields the most complete silhouettes of the moving taxi and truck, frame differencing yields only their leading and trailing edges, and optical flow is the only method that reports direction and speed. On average 1.72 % of pixels are flagged by differencing and 2.57 % by GMM (peaks of 17 % and 22 % when large vehicles cross the frame); the mean LK displacement is 1.19 px/frame with a 90th-percentile of 3.7 px.")

H("7. Overall Conclusion")
B(["Restoration: Gaussian filter is best for Gaussian noise (24.6 dB), median for salt-and-pepper (26.7 dB, SSIM 0.89).",
   "Segmentation: Otsu + opening/closing gives a clean binary mask; watershed on the distance transform separated overlapping coins that connected-component labelling merged (2 → 3 objects).",
   "Features: SIFT matched two different views with 96 % RANSAC inliers and stayed stable under rotation and 50 % down-scaling.",
   "Recognition: YOLOv8 greatly outperformed the Haar cascade on a real street scene, but still produced wrong labels on an out-of-distribution scene (idol pandal).",
   "Motion: the three motion methods are complementary – GMM for silhouettes, differencing for fast cheap cues, Lucas–Kanade for direction and magnitude."])
H("8. Limitations")
B(["The recorded video is 13.8 s (shorter than the 15 s suggested) and hand-held, which increases false motion.",
   "Haar and YOLO are used with default pretrained weights and thresholds; no fine-tuning was done."])

H("9. How to Reproduce (Step-by-Step)")
for s in ["Install Python 3.10+ and create a virtual environment: python -m venv venv && source venv/bin/activate (Windows: venv\\Scripts\\activate).",
          "Install dependencies: pip install -r requirements.txt.",
          "Place the inputs in data/: durga_1.jpg, durga_2.jpg, coins.jpg and video.mp4 (video is hosted on Google Drive – link in README).",
          "Download the Haar cascade XML files haarcascade_frontalface_default.xml and haarcascade_fullbody.xml from the OpenCV GitHub repository into data/haarcascades/. YOLO weights (yolov8n.pt) download automatically on first run.",
          "Run everything: python src/run_all.py (or run each task script separately). Outputs are written to outputs/task1 … task4.",
          "Build this report: python docs/build_report.py."]:
    d.add_paragraph(s, style="List Number")

H("10. Source Code Listing")
P("Full code is in the Git repository and reproduced below.")
for f in ["common.py", "task1_restoration_segmentation.py", "task2_features.py", "task3_detection.py", "task4_motion.py", "run_all.py"]:
    H(f"src/{f}", 2); p = d.add_paragraph(); r = p.add_run(open(os.path.join(R, "src", f)).read()); r.font.name = "Consolas"; r.font.size = Pt(7.5)
d.save(os.path.join(R, "docs", "Technical_Report.docx")); print("saved")
