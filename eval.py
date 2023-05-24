import os
import random

import torch
import numpy as np
import taichi as ti

from gui import NGPGUI
from opt import get_opts
from datasets import dataset_dict


def main():
    seed = 23
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    hparams = get_opts()

    val_dir = "results/"
    assert os.path.exists(val_dir)
    hparams.ckpt_path = os.path.join(val_dir, "model.pth")

    dataset = dataset_dict[hparams.dataset_name](
        root_dir=hparams.root_dir,
        downsample=hparams.downsample,
        read_meta=True,
    )
    model_config = {
        "scale": hparams.scale,
        "pos_encoder_type": hparams.encoder_type,
        "max_res": 1024 if hparams.scale == 0.5 else 4096,
        "half_opt": hparams.half_opt,
    }

    ti.init(arch=ti.cuda)

    NGPGUI(hparams, model_config, dataset.K, dataset.img_wh, dataset.poses).render()


if __name__ == "__main__":
    main()
