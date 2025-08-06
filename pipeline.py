import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ─── CONFIG ─────────────────────────────────────────────────────────────
PERSON_PATH = 'data/person1.jpg'
CLOTH_PATH  = 'data/cloth.png'
OUT_DIR     = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# Load YOLOv8-Pose
pose_model = YOLO('yolov8n-pose.pt')  # auto-download weights

def save(img: np.ndarray, name: str):
    """Save a NumPy image to outputs/"""
    Image.fromarray(img).save(os.path.join(OUT_DIR, name))

# ─── STAGE 0: Save inputs ────────────────────────────────────────────────
person = Image.open(PERSON_PATH).convert('RGB')
cloth  = Image.open(CLOTH_PATH).convert('RGBA')
p_np   = np.array(person)
c_np   = np.array(cloth)
save(p_np, 'stage0_person.jpg')
save(c_np, 'stage0_cloth.png')

# ─── STAGE 1: Dummy parse mask ───────────────────────────────────────────
# (white mask everywhere; swap in a real human parser for production)
mask = np.ones((p_np.shape[0], p_np.shape[1]), dtype=np.uint8) * 255
save(mask, 'stage1_parse.png')

# ─── STAGE 2: Perspective warp of cloth ─────────────────────────────────
# 1) Detect shoulder & hip keypoints
res = pose_model.predict(p_np, imgsz=640, device='cpu', verbose=False)[0]
kpts = res.keypoints
if kpts is None or kpts.xy is None or len(kpts.xy)==0:
    raise RuntimeError("No person detected!")
xy = kpts.xy[0].cpu().numpy().astype(int)

# YOLO indices: 5=left shoulder, 6=right shoulder, 12=right hip, 11=left hip
tl = tuple(xy[5])   # left shoulder → top-left of warp target
tr = tuple(xy[6])   # right shoulder → top-right
br = tuple(xy[12])  # right hip     → bottom-right
bl = tuple(xy[11])  # left hip      → bottom-left

# 2) Compute 4-point transform
h_p, w_p = p_np.shape[:2]
h_c, w_c = c_np.shape[:2]
src = np.array([[0,0],[w_c-1,0],[w_c-1,h_c-1],[0,h_c-1]], dtype=np.float32)
dst = np.array([tl, tr, br, bl], dtype=np.float32)
M   = cv2.getPerspectiveTransform(src, dst)

# 3) Warp cloth to person frame size
warped = cv2.warpPerspective(
    c_np, M, 
    (w_p, h_p),                       # width, height
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0,0,0,0)             # transparent outside quad
)
save(warped, 'stage2_warped_cloth.png')

# ─── STAGE 3: Grid visualization ────────────────────────────────────────
# Draw red grid on cloth, then warp it
grid = np.zeros_like(c_np)
step = min(w_c, h_c)//16
for y in range(0, h_c, step):
    cv2.line(grid, (0,y), (w_c,y), (0,0,255,255), 1)
for x in range(0, w_c, step):
    cv2.line(grid, (x,0), (x,h_c), (0,0,255,255), 1)
grid_vis = cv2.warpPerspective(
    grid, M, (w_p, h_p),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0,0,0,0)
)
save(grid_vis, 'stage3_grid.png')

# ─── STAGE 4: Final composite ───────────────────────────────────────────
alpha   = warped[:,:,3:4] / 255.0            # (h_p,w_p,1)
cloth3  = warped[:,:,:3].astype(np.float32)  # (h_p,w_p,3)
base3   = p_np.astype(np.float32)            # (h_p,w_p,3)
comp    = (cloth3 * alpha) + (base3 * (1-alpha))
save(comp.astype(np.uint8), 'stage4_final.png')

print("✅ Done! Check outputs/ for stage0–4 images.")
