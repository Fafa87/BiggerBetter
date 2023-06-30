import os
import pathlib

import attrdict
import fire
import imageio

import foreign.autoencoder.autoencoder
from bb.utils import *


def train_autoencoder(config, data_root, output_dir):
    train_config = attrdict.AttrDict()
    train_config.data_root = data_root
    os.makedirs(output_dir, exist_ok=True)
    train_config.log_dir = output_dir / "logs"

    train_config.num_workers = config.num_workers
    train_config.image_size = config.image_size
    train_config.max_epochs = config.max_epochs
    train_config.batch_size = config.batch_size
    train_config.train_val_test_split = config.train_val_test_split

    train_config.nc = config.nc
    train_config.nz = config.nz
    train_config.nfe = config.nfe
    train_config.nfd = config.nfd
    train_config.lr = config.lr
    train_config.beta1 = config.beta1
    train_config.beta2 = config.beta2
    train_config.gpus = config.gpus

    foreign.autoencoder.autoencoder.main(train_config)


def train(config_name, data_root, output_root):
    data_root = pathlib.Path(data_root)
    output_root = pathlib.Path(output_root)
    config = load_config(config_name)
    config = attrdict.AttrDict(config)

    if config.type == "autoencoder":
        train_autoencoder(config, data_root, output_root)
    else:
        raise NotImplementedError(config.type)


if __name__ == "__main__":
    fire.Fire(train)
