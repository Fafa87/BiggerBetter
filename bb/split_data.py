import glob
import shutil
from pathlib import Path

import fire
import imageio

import numpy as np


def split_train_val_to_div2k(low_res_dir, high_res_dir, div2k_output):
    low_res_dir = Path(low_res_dir)
    high_res_dir = Path(high_res_dir)

    low_res_data = sorted(low_res_dir.rglob("*.*"))
    high_res_data = sorted(high_res_dir.rglob("*.*"))

    low_to_res = list(zip(low_res_data, high_res_data))
    np.random.shuffle(low_to_res)

    count = len(low_to_res)
    print(f"Samples count: {count}")
    train = low_to_res[:int(count * 0.8)]
    val = low_to_res[int(count * 0.8):]

    div2k_output = Path(div2k_output)
    train_root = div2k_output / "DIV2K_train_HR"
    for low, high in train:
        shutil.copy(low, train_root / low.name)
    train_root = div2k_output / "DIV2K_val_HR"
    for low, high in val:
        shutil.copy(low, train_root / low.name)

    train_root = div2k_output / "DIV2K_train_LR_unknown" / "X2"
    for low, high in train:
        shutil.copy(low, train_root / low.name)
    train_root = div2k_output / "DIV2K_val_LR_unknown" / "X2"
    for low, high in val:
        shutil.copy(low, train_root / low.name)


if __name__ == "__main__":
    fire.Fire()
