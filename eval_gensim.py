import torch
import numpy as np
import gym
import gym.spaces
import os
import hydra
import logging
import wandb
import yaml
from omegaconf import OmegaConf

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from tensordict import TensorDict, tensorclass, MemoryMappedTensor

from models import WorldModel
from pprint import pprint

from torchvision.io import write_video
from torchvision.utils import make_grid
from torchvision.transforms.functional import resize
from gen_diversity.dataset.gensim import GensimDataset

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
CFG_PATH = os.path.join(FILE_PATH, "cfg")
GENSIM_ROOT = os.environ.get("GENSIM_ROOT")
IMAGE_SIZE = (128, 128)

if GENSIM_ROOT is None:
    raise ValueError("Please set GENSIM_ROOT environment variable")


# get all available evaluations
HYDRA_OUTPUT_PATH = os.path.join(FILE_PATH, "outputs")
RUN_DIRS = []
for date in os.listdir(HYDRA_OUTPUT_PATH):
    for run_time in os.listdir(os.path.join(HYDRA_OUTPUT_PATH, date)):
        run_dir = os.path.join(HYDRA_OUTPUT_PATH, date, run_time)
        if os.path.exists(os.path.join(run_dir, "wandb", "latest-run", "files", "ckpt-9.pt")):
            RUN_DIRS.append(run_dir)

for run_dir in RUN_DIRS:
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    config = yaml.safe_load(open(config_path))
    ckpt_path = os.path.join(run_dir, "wandb", "latest-run", "files", "ckpt-9.pt")
    print(config["dataset"], config["max_episodes"], ckpt_path)


@hydra.main(config_path=CFG_PATH, config_name="eval")
def main(cfg):
    device = cfg.world_model.device

    data_dir = os.path.join(GENSIM_ROOT, "data", "train", cfg.dataset)

    try:
        dataset = GensimDataset.load(data_dir, 40, high_level=False, max_episodes=40)
    except Exception as e:
        print(f"Failed to load dataset due to {e}, creating new one")
        dataset = GensimDataset.make(data_dir, 40, high_level=False, max_episodes=40)
    

    model = WorldModel(
        obs_space=gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, (*IMAGE_SIZE, 3), dtype=np.uint8),
                "state": gym.spaces.Box(0, 20, (12,), dtype=np.float32),
            }
        ),
        act_space=gym.spaces.Box(-1, 1, (6,), dtype=np.float32),
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
    
    dataloader = DataLoader(
        dataset, 
        batch_size=32,
        shuffle=True, 
        collate_fn=torch.stack
    )

    model.eval()
    losses_all = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            # some manual processing
            data = data.to(device)
            data["image"] = resize(data["image"].flatten(0, 1), IMAGE_SIZE).unflatten(0, data.shape)
            data["image"] = data["image"].permute(0, 1, 3, 4, 2)
            data["image"] = data["image"][..., :3] / 255.0
            data["cont"] = (1.0 - data["is_terminal"].float()).unsqueeze(-1)
            data["is_first"] = data["is_first"].float()

            losses = model._eval(data)
            losses = {k: torch.mean(v).item() for k, v in losses.items()}
            losses_all.append(TensorDict(losses, []))
    losses_all = torch.stack(losses_all)
    losses_all = {k: torch.mean(v).item() for k, v in losses_all.items()}
    
    os.makedirs("eval", exist_ok=True)
    yaml.dump(losses_all, open(f"eval/{cfg.dataset}.yaml", "w"))
    pprint(losses_all)

if __name__ == "__main__":
    main()
    