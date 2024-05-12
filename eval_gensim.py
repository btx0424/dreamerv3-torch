import torch
import numpy as np
import gym
import gym.spaces
import os
import hydra
import logging
import wandb
from omegaconf import OmegaConf

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from tensordict import TensorDict, tensorclass, MemoryMappedTensor

from models import WorldModel
from pprint import pprint

from torchvision.io import write_video
from torchvision.utils import make_grid
from sentence_transformers import SentenceTransformer

from dataset.gensim import GensimDataset

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
CFG_PATH = os.path.join(FILE_PATH, "cfg")

@hydra.main(config_path=CFG_PATH, config_name="eval")
def main(cfg):
    device = cfg.world_model.device

    model = WorldModel(
        obs_space=gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, (240, 320, 3), dtype=np.uint8),
                "state": gym.spaces.Box(0, 20, (12,), dtype=np.float32),
            }
        ),
        act_space=None,
        step=0,
        config=cfg.world_model,
    ).to(device)

    # language_sim = True
    # if language_sim:
    #     llm = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./llm")

    if cfg.get("ckpt", None) is not None:
        print(f"load checkpoint from {cfg.ckpt}")
        ckpt = torch.load(cfg.ckpt)
        model.load_state_dict(ckpt)
        del ckpt
    
    train_dataset = GensimDataset.make("/localdata/bxu/isaac_ws/GenEval/GenSim/data/train")
    print(train_dataset.task_desc)

    try:
        dataset = GensimDataset.load(cfg.dataset.path, 40)
    except:
        dataset = GensimDataset.make(cfg.dataset.path, 40)
    
    model.eval()
    print(dataset.tasks)
    task_losses = {}
    with torch.no_grad():
        for task in dataset.tasks:
            print(dataset.task_desc[task])
            losses_all = []
            t = tqdm(dataset.episodes(task), total=len(dataset.task_episodes[task]), desc=task)
            for i, data in enumerate(t):
                # some manual processing
                data = data.to(device).unsqueeze(0)
                data["image"] = data["image"][..., :3] / 255.0
                data["cont"] = (1.0 - data["is_terminal"]).unsqueeze(-1)

                losses = model._eval(data)
                losses = {k: torch.mean(v).item() for k, v in losses.items()}
                losses_all.append(TensorDict(losses, []))
            losses_all = torch.stack(losses_all)
            losses_all = {k: torch.mean(v).item() for k, v in losses_all.items()}
            task_losses[task] = losses_all
            pprint(losses_all)
    pprint(task_losses)

if __name__ == "__main__":
    main()
    