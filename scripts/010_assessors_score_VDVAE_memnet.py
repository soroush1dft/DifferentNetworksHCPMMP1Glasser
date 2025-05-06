#!/usr/bin/env python3
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Updated MemNet path
sys.path.append('C:/Users/sOrOush/SoroushProjects/01_Soroush_and_Shakiba/NSD_High_Dimensional_Data/11_Marco_And_Soroush/Scripts/GANalyze/pytorch')
from assessors import memnet

# Load mean image for preprocessing (updated path)
mean = np.load('C:/Users/sOrOush/SoroushProjects/01_Soroush_and_Shakiba/NSD_High_Dimensional_Data/11_Marco_And_Soroush/Data/GANalyze/pytorch/assessors/image_mean.npy')

# -----------------------------------------------------------------------------
# Image-loading and preprocessing
# -----------------------------------------------------------------------------
def prepare_images_folder_memnet(folder: Path, target_size=(256, 256), mean=None) -> torch.Tensor:
    def natural_key(s):
        import re
        return [int(c) if c.isdigit() else c.lower()
                for c in re.split(r'(\d+)', s)]

    files = sorted(
        [f for f in folder.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')],
        key=lambda p: natural_key(p.name)
    )

    transform = T.Compose([
        T.Resize(target_size),
        T.Lambda(lambda x: np.array(x)),
        T.Lambda(lambda x: np.subtract(x[:, :, [2, 1, 0]], mean) if mean is not None else x),
        T.Lambda(lambda x: x[15:242, 15:242]),
        T.ToTensor()
    ])

    tensors = []
    for f in files:
        img = Image.open(f).convert('RGB')
        tensors.append(transform(img))

    return torch.stack(tensors)

# -----------------------------------------------------------------------------
# Scoring utility for MemNet
# -----------------------------------------------------------------------------
def score_batch_memnet(assessor, batch: torch.Tensor, device='cuda'):
    batch = batch.to(device)
    with torch.no_grad():
        scores = assessor(batch)
    return [s.detach().cpu().numpy()[0] for s in scores]

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    real_images_dir = Path(r"C:\Users\sOrOush\SoroushProjects\14_CLIP_Ozcelic\results\00_Original_Images")
    generated_base = Path(r"C:\Users\sOrOush\SoroushProjects\14_CLIP_Ozcelic\results\generated images\VDVAE")
    output_pkl = Path(r"C:\Users\sOrOush\SoroushProjects\14_CLIP_Ozcelic\results\assessor_results\memnet_res.pkl")

    groups = ['nsdgeneral', 'Default', 'Auditory']
    degrees = [1, 2, 3, 4]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assessor = memnet.MemNet().to(device).eval()

    results = {}

    print("Scoring real images...")
    real_batch = prepare_images_folder_memnet(real_images_dir, mean=mean)
    results['real_img'] = score_batch_memnet(assessor, real_batch, device)
    print("  → real images done.")

    for grp in groups:
        for deg in degrees:
            key = f"{grp}_degree{deg}"
            folder = generated_base / grp / f"degree{deg}" / "subj01"
            print(f"Scoring {key}…", end='')

            if not folder.exists():
                print("  SKIPPED (folder not found)")
                results[key] = []
                continue

            batch = prepare_images_folder_memnet(folder, mean=mean)
            results[key] = score_batch_memnet(assessor, batch, device)
            print("  done.")

    print(f"Saving results to {output_pkl}")
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(results, f)

    print("All done!")

if __name__ == "__main__":
    main()

