#!/usr/bin/env python3
import os
import sys
import glob
import json

import cv2
import torch
from PIL import Image
import torchvision.transforms.functional as TF

# —————————————————————————————————————————
# 1) Point to your assessor code and import it
# —————————————————————————————————————————
ASSESSOR_PATH = r"C:\Users\sOrOush\SoroushProjects\01_Soroush_and_Shakiba\NSD_High_Dimensional_Data\11_Marco_And_Soroush\Scripts\GANalyze\pytorch\assessors"
sys.path.append(ASSESSOR_PATH)

from emonet import emonet

def load_assessor(tencrop: bool = False) -> torch.nn.Module:
    """Load EmoNet, freeze it, set to eval, move to GPU if available."""
    model, _, _ = emonet(tencrop=tencrop)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

# —————————————————————————————————————————
# 2) Preprocessing: read one image, resize & to‐tensor
# —————————————————————————————————————————
def preprocess_image(path: str) -> torch.Tensor:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    pil = TF.resize(pil, (256, 256))
    return TF.to_tensor(pil)

# —————————————————————————————————————————
# 3) Score all PNGs in a folder, one by one
# —————————————————————————————————————————
def score_folder(assessor: torch.nn.Module, folder: str) -> list[float]:
    device = next(assessor.parameters()).device
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    scores = []
    for p in paths:
        tensor = preprocess_image(p).unsqueeze(0).to(device)
        with torch.no_grad():
            out = assessor(tensor)            # returns a list of 1 tensor
        score = out[0].cpu().item()          # scalar
        scores.append(score)
    return scores

# —————————————————————————————————————————
# 4) Main orchestration
# —————————————————————————————————————————
def main():
    assessor = load_assessor(tencrop=False)

    # Where to write per‐folder JSONs:
    RESULT_DIR = r"C:\Users\sOrOush\SoroushProjects\14_CLIP_Ozcelic\results\assessor_results"
    os.makedirs(RESULT_DIR, exist_ok=True)

    # A) Real images
    REAL_DIR = r"C:\Users\sOrOush\SoroushProjects\14_CLIP_Ozcelic\results\00_Original_Images"
    real_scores = score_folder(assessor, REAL_DIR)
    out_path = os.path.join(RESULT_DIR, "real_scores.json")
    with open(out_path, "w") as f:
        json.dump(real_scores, f)
    print(f"Saved {len(real_scores)} real-image scores → {out_path}")

    # B) VDVAE‐generated images, only degrees 1 & 2
    VDVAE_DIR = r"C:\Users\sOrOush\SoroushProjects\14_CLIP_Ozcelic\results\generated images\VDVAE"
    groups = ["nsdgeneral", "Default", "Auditory"]
    degrees = [1, 2]
    subject = "subj01"

    for grp in groups:
        for deg in degrees:
            folder = os.path.join(VDVAE_DIR, grp, f"degree{deg}", subject)
            if not os.path.isdir(folder):
                print(f"[WARN] folder not found: {folder}")
                continue
            scores = score_folder(assessor, folder)
            fname = f"{grp}_degree{deg}_scores.json"
            save_to = os.path.join(RESULT_DIR, fname)
            with open(save_to, "w") as f:
                json.dump(scores, f)
            print(f"Saved {len(scores)} scores for {grp} degree {deg} → {save_to}")

if __name__ == "__main__":
    main()
