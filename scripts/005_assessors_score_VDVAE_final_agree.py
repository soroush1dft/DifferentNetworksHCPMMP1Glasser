#!/usr/bin/env python3
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from emonet import emonet  # assumes emonet is on your PYTHONPATH

# -----------------------------------------------------------------------------
# 1) Load and freeze EmoNet assessor
# -----------------------------------------------------------------------------
def load_assessor(device: str = 'cpu', tencrop: bool = False):
    model, input_transform, output_transform = emonet(tencrop=tencrop)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, input_transform, output_transform

# -----------------------------------------------------------------------------
# 2) Image‑loading helper for a folder of .png/.jpg files
# -----------------------------------------------------------------------------
def prepare_images_folder(folder: Path, size=(256, 256)) -> torch.Tensor:
    """
    Reads all images in `folder`, sorts them naturally,
    resizes to `size`, converts to a float Tensor in [0,1],
    and stacks into a single batch Tensor of shape (N, 3, H, W).
    """
    def natural_key(s):
        import re
        return [int(c) if c.isdigit() else c.lower()
                for c in re.split(r'(\d+)', s)]

    files = sorted(
        [f for f in folder.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')],
        key=lambda p: natural_key(p.name)
    )

    tensors = []
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor()
    ])
    for f in files:
        img = Image.open(f).convert('RGB')
        tensors.append(transform(img))
    return torch.stack(tensors, dim=0)

# -----------------------------------------------------------------------------
# 3) Batch scoring utility
# -----------------------------------------------------------------------------
def score_batch(model, batch: torch.Tensor, input_transform, output_transform, device: str = 'cpu'):
    """
    Applies optional input_transform, runs the model, applies optional output_transform,
    and returns a list of numpy arrays (one per image).
    """
    batch = batch.to(device)
    if input_transform is not None:
        batch = input_transform(batch)
    with torch.no_grad():
        outputs = model(batch)
    results = []
    for out in outputs:
        if output_transform is not None:
            out = output_transform(out)
        results.append(out.cpu().numpy())
    return results

# -----------------------------------------------------------------------------
# 4) Main entrypoint
# -----------------------------------------------------------------------------
def main():
    # Paths (customize these as needed)
    real_images_dir = Path(r"C:\Users\sOrOush\SoroushProjects\14_CLIP_Ozcelic\results\00_Original_Images")
    generated_base = Path(r"C:\Users\sOrOush\SoroushProjects\14_CLIP_Ozcelic\results\generated images\VDVAE")
    output_pkl    = Path(r"C:\Users\sOrOush\SoroushProjects\14_CLIP_Ozcelic\results\assessor_results\emonet_res.pkl")

    # Define your groups and degrees
    groups = ['nsdgeneral', 'Default', 'Auditory']
    degrees = [1, 2, 3, 4]

    # Choose device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assessor, inp_tf, out_tf = load_assessor(device=device, tencrop=False)

    # Result dict
    emonet_res = {}

    # 4.1) Score real images
    print("Scoring real images…")
    real_batch = prepare_images_folder(real_images_dir)
    emonet_res['real_img'] = score_batch(assessor, real_batch, inp_tf, out_tf, device)
    print("  → real images done.")

    # 4.2) Score generated images by group/degree
    for grp in groups:
        for deg in degrees:
            key = f"{grp}_degree{deg}"
            folder = generated_base / grp / f"degree{deg}" / "subj01"
            print(f"Scoring {key}…", end='')
            if not folder.exists():
                print("  SKIPPED (folder not found)")
                emonet_res[key] = []
                continue

            batch = prepare_images_folder(folder)
            emonet_res[key] = score_batch(assessor, batch, inp_tf, out_tf, device)
            print("  done.")

    # 5) Save results
    print(f"Saving results to {output_pkl}")
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(emonet_res, f)

    print("All done!")

if __name__ == "__main__":
    main()
