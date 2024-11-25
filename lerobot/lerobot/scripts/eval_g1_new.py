#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
python lerobot/scripts/eval.py -p lerobot/diffusion_pusht eval.n_episodes=10
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.

```
python lerobot/scripts/eval.py \
    -p outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    eval.n_episodes=10
```

Note that in both examples, the repo/folder should contain at least `config.json`, `config.yaml` and
`model.safetensors`.

Note the formatting for providing the number of episodes. Generally, you may provide any number of arguments
with `qualified.parameter.name=value`. In this case, the parameter eval.n_episodes appears as `n_episodes`
nested under `eval` in the `config.yaml` found at
https://huggingface.co/lerobot/diffusion_pusht/tree/main.
"""

import argparse
import json
import logging
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path
from typing import Callable

import einops
import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import Subset, Sampler
from datasets import Dataset, Features, Image, Sequence, Value, concatenate_datasets
from huggingface_hub import snapshot_download
from huggingface_hub.utils._errors import RepositoryNotFoundError

# from huggingface_hub import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from PIL import Image as PILImage
from torch import Tensor, nn
from tqdm import trange

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import hf_transform_to_torch
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.logger import log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed
import cv2

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
# print(f"project_root:{project_root}")
sys.path.append(str(project_root))
from unitree_utils.image_server.image_client import ImageClient
from unitree_utils.robot_control.robot_arm import G1_29_ArmController
from unitree_utils.robot_control.robot_hand_unitree import Dex3_1_Controller
from multiprocessing import Process, shared_memory, Array

import pdb
from tqdm import tqdm

def get_image_processed(cam, img_size=[640, 480]):
    # realsense return cv2 image, BGR format
    curr_images = []
    color_img  = cam
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    color_img = cv2.resize(color_img, img_size)         # w, h
    curr_images.append(color_img)
    color_img = np.stack(curr_images, axis=0)
    return color_img

class RandomSampler(Sampler):
    def __init__(self,dataset,fraction=0.1,seed=None):
        self.dataset = dataset
        self.num_samples = int(len(dataset)*fraction)
        self.seed = seed
    
    def __iter__(self):
        if self.seed is not None:
            np.random.choice(self.seed)
        else:
            np.random.seed(self.seed)
        indices = np.random.choice(range(len(self.dataset)),self.num_samples,replace=False)
        # pdb.set_trace()
        yield from iter(indices.tolist())

    def __len__(self):
        return self.num_samples

def eval_policy(
    # env,
    policy: torch.nn.Module,
    dataset_loader: torch.utils.data.DataLoader,
) -> dict:
    """
    Evaluate the policy using dataset samples.

    Args:
        policy: The PyTorch model to evaluate.
        dataset_loader: DataLoader providing the evaluation dataset.

    Returns:
        A dictionary of evaluation metrics.
    """
    device = get_device_from_parameters(policy)
    policy.eval()
    all_loss = []
    for batch in tqdm(dataset_loader):
        batch = {key:batch[key].to(device) for key in batch.keys()}
        # pdb.set_trace()
        output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
        loss = output_dict["loss"]
        all_loss.append(loss.item())

    # pdb.set_trace()
    all_loss = np.array(all_loss)
    loss_avg = all_loss.mean()
    loss_var = all_loss.std()
    info = {
        "loss_avg": loss_avg.item(),
        "loss_var": loss_var.item()
    }

    return info

def main(
    pretrained_policy_path: Path | None = None,
    hydra_cfg_path: str | None = None,
    out_dir: str | None = None,
    config_overrides: list[str] | None = None,
):
    assert (pretrained_policy_path is None) ^ (hydra_cfg_path is None)
    if pretrained_policy_path is not None:
        hydra_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"), config_overrides)
    else:
        hydra_cfg = init_hydra_config(hydra_cfg_path, config_overrides)

    if out_dir is None:
        out_dir = f"outputs/eval/{dt.now().strftime('%Y-%m-%d/%H-%M-%S')}_{hydra_cfg.env.name}_{hydra_cfg.policy.name}"

    # Check device is available
    device = get_safe_torch_device(hydra_cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # set_global_seed(hydra_cfg.seed)

    log_output_dir(out_dir)

    logging.info("Making environment.")
    env = make_env(hydra_cfg)

    logging.info("Making policy.")
    if hydra_cfg_path is None:
        policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_policy_path))
    else:
        # Note: We need the dataset stats to pass to the policy's normalization modules.
        policy = make_policy(hydra_cfg=hydra_cfg, dataset_stats=make_dataset(hydra_cfg).stats)

    assert isinstance(policy, nn.Module)
    policy.eval()

    device = get_safe_torch_device(hydra_cfg.device, log=True) 
    offline_dataset = make_dataset(hydra_cfg)
    # indices = np.random.choice(range(0,len(offline_dataset)),size=len(offline_dataset)//10,replace=False)
    # sub_offline_dataset = Subset(offline_dataset,indices)
    # pdb.set_trace()
    random_sampler = RandomSampler(offline_dataset)
    dataloader = torch.utils.data.DataLoader(
        offline_dataset,
        num_workers=4,
        batch_size=32,
        shuffle=False,
        sampler=random_sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    # pdb.set_trace()


    with torch.no_grad(), torch.autocast(device_type=device.type) if hydra_cfg.use_amp else nullcontext():
        info = eval_policy(
            # env,
            policy,
            dataloader
        )
    print(info)

    # Save info
    # with open(Path(out_dir) / "eval_info.json", "w") as f:
    #     json.dump(info, f, indent=2)

    # env.close()

    # logging.info("End of eval")


if __name__ == "__main__":
    init_logging()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
            "(useful for debugging). This argument is mutually exclusive with `--config`."
        ),
       default='/home/unitree/Videos/lw/21-53-27_real_world_act_default/checkpoints/100000/pretrained_model'
    )
    group.add_argument(
        "--config",
        help=(
            "Path to a yaml config you want to use for initializing a policy from scratch (useful for "
            "debugging). This argument is mutually exclusive with `--pretrained-policy-name-or-path` (`-p`)."
        ),
    )
    parser.add_argument("--revision", help="Optionally provide the Hugging Face Hub revision ID.")
    parser.add_argument(
        "--out-dir",
        help=(
            "Where to save the evaluation outputs. If not provided, outputs are saved in "
            "outputs/eval/{timestamp}_{env_name}_{policy_name}"
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

    if args.pretrained_policy_name_or_path is None:
        main(hydra_cfg_path=args.config, out_dir=args.out_dir, config_overrides=args.overrides)
    else:
        try:
            pretrained_policy_path = Path(
                snapshot_download(args.pretrained_policy_name_or_path, revision=args.revision)
            )
        except (HFValidationError, RepositoryNotFoundError) as e:
            if isinstance(e, HFValidationError):
                error_message = (
                    "The provided pretrained_policy_name_or_path is not a valid Hugging Face Hub repo ID."
                )
            else:
                error_message = (
                    "The provided pretrained_policy_name_or_path was not found on the Hugging Face Hub."
                )

            logging.warning(f"{error_message} Treating it as a local directory.")
            pretrained_policy_path = Path(args.pretrained_policy_name_or_path)
        if not pretrained_policy_path.is_dir() or not pretrained_policy_path.exists():
            raise ValueError(
                "The provided pretrained_policy_name_or_path is not a valid/existing Hugging Face Hub "
                "repo ID, nor is it an existing local directory."
            )

        main(
            pretrained_policy_path=pretrained_policy_path,
            out_dir=None,
            config_overrides=None,
        )
