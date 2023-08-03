import os
from pathlib import Path

import imageio
import torch
import fire

from torchsr.datasets import Div2K
from torchsr.models import ninasr_b0, ninasr_b1, rcan
from torchvision.transforms.functional import to_pil_image, to_tensor

import torchvision.transforms.functional as F

def to_image(t):
    """Workaround a bug in torchvision
    The conversion of a tensor to a PIL image causes overflows, which result in huge errors"""
    if t.ndim == 4:
        t = t.squeeze(0)
    t = t.mul(255).round().div(255).clamp(0, 1)
    return F.to_pil_image(t.cpu())


def run_on(sample_id=None, sample_paths=None, model_path=None, output_root=None):
    output_root = Path(output_root)
    os.makedirs(output_root, exist_ok=True)
    if sample_id is not None:
        # Div2K dataset
        dataset = Div2K(root="./data", scale=2, download=True)

        # Get the first image in the dataset (High-Res and Low-Res)
        hr, lr = dataset[sample_id]
        lrs = [(str(sample_id), lr)]
    if sample_paths is not None:
        if isinstance(sample_paths, str):
            sample_paths = [sample_paths]
        sample_paths = [Path(s) for s in sample_paths]
        lrs = [(p.stem, imageio.v3.imread(p)) for p in sample_paths]

    # Download a pretrained NinaSR model
    model = ninasr_b0(scale=2, pretrained=True)
    if model_path:
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=map_location)
        model.load_state_dict(checkpoint['state_dict'])

    for s, lr in lrs:
        # Run the Super-Resolution model
        lr_t = to_tensor(lr).unsqueeze(0)
        sr_t = model(lr_t)
        sr = to_image(sr_t)
        sr.save((output_root / s).with_suffix(".png"))
        #sr.show()


if __name__ == "__main__":
    fire.Fire(run_on)
