from typing import Dict
import json

import numpy as np
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import torchvision.transforms.v2 as tv_v2

from flower.models.flower import FLOWERVLA

if __name__ == "__main__":
    episode_data_calvin = np.load(
        # r"C:\Users\mohes\Documents\Coding\flower_vla_calvin\data\calvin_debug_dataset\calvin_debug_dataset\validation\episode_0553567.npz"
        "/home/moritz/Documents/Repos/flower_vla_calvin/data/calvin_debug_dataset/validation/episode_0553568.npz"
    )
    episode_data_calvin = dict(episode_data_calvin)

    obs = {
        "rgb_obs": {
            # B, T, C, H, W
            "rgb_static": tv_v2.Resize((224, 224))(
                torch.from_numpy(episode_data_calvin["rgb_static"]).permute(2, 1, 0)[
                    None, None
                ]
            ),
            "rgb_gripper": tv_v2.Resize((224, 224))(
                torch.from_numpy(episode_data_calvin["rgb_gripper"]).permute(2, 1, 0)[
                    None, None
                ]
            ),
        }
    }

    print("loading model")
    flower_model_basic = FLOWERVLA()

    print("sepping ...")
    output = flower_model_basic.step(obs, goal={"lang_text": "pick up the blue cube"})

    print(output)
    print("done")
