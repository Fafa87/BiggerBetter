import glob
import os
import shutil
from pathlib import Path

import fire
import imageio

import numpy as np
from tqdm import tqdm


def split_train_val_to_div2k(low_res_dir, high_res_dir, div2k_output, clean=False):
    low_res_dir = Path(low_res_dir)
    high_res_dir = Path(high_res_dir)

    low_res_data = list(low_res_dir.rglob("*.*"))
    high_res_data = list(high_res_dir.rglob("*.*"))

    low_res_names = set([f.stem for f in low_res_data])
    high_res_names = set([f.stem for f in high_res_data])
    good_names = low_res_names & high_res_names

    low_res_data = [a for a in low_res_data if a.stem in good_names]
    high_res_data = [a for a in high_res_data if a.stem in good_names]

    low_to_res = list(zip(low_res_data, high_res_data))
    np.random.shuffle(low_to_res)
    # TMP
    #low_to_res = low_to_res[:1000]
    count = len(low_to_res)
    print(f"Samples count: {count}")
    train = low_to_res[:int(count * 0.9)]
    val = low_to_res[int(count * 0.9):]

    div2k_output = Path(div2k_output)
    train_root = div2k_output / "DIV2K_train_HR"
    if train_root.exists() and clean:
        print("Removing tree...")
        shutil.rmtree(div2k_output)
    os.makedirs(train_root, exist_ok=True)

    for low, high in tqdm(train):
        if not (train_root / low.name).is_file():
            shutil.copyfile(high, train_root / low.name)
    train_root = div2k_output / "DIV2K_valid_HR"
    os.makedirs(train_root, exist_ok=True)
    for low, high in val:
        if not (train_root / low.name).is_file():
            shutil.copy(high, train_root / low.name)

    train_root = div2k_output / "DIV2K_train_LR_unknown" / "X2"
    os.makedirs(train_root, exist_ok=True)
    for low, high in tqdm(train):
        if not (train_root / low.name).is_file():
            shutil.copy(low, train_root / low.name)
    train_root = div2k_output / "DIV2K_valid_LR_unknown" / "X2"
    os.makedirs(train_root, exist_ok=True)
    for low, high in val:
        if not (train_root / low.name).is_file():
            shutil.copy(low, train_root / low.name)


if __name__ == "__main__":
    fire.Fire()
