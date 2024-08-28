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
from torchvision.transforms.functional import resize
from gen_diversity.dataset.gensim import GensimDataset

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
GENSIM_ROOT = os.environ.get("GENSIM_ROOT")
IMAGE_SIZE = (128, 128)

if GENSIM_ROOT is None:
    raise ValueError("Please set GENSIM_ROOT environment variable")


@hydra.main(config_path=os.path.join(FILE_PATH, "cfg"), config_name="train")
def main(cfg):
    device = cfg.world_model.device

    run = wandb.init(
        project="GensimEval", 
        entity="btx0424",
        mode=cfg.wandb.mode,
    )
    run.config["project"] = "gensim"

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

    if cfg.get("ckpt", None) is not None:
        print(f"load checkpoint from {cfg.ckpt}")
        ckpt = torch.load(cfg.ckpt)
        model.load_state_dict(ckpt)
        del ckpt


    data_dir = os.path.join(GENSIM_ROOT, "data", "train", cfg.dataset)

    try:
        dataset = GensimDataset.load(data_dir, 40, high_level=False, max_episodes=cfg.max_episodes)
    except:
        dataset = GensimDataset.make(data_dir, 40, high_level=False, max_episodes=cfg.max_episodes)
    

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        collate_fn=torch.stack
    )

    for epoch in range(cfg.get("epochs", 10)):
        for i, data in enumerate(tqdm(dataloader)):
            # some manual processing
            data = data.to(device)
            data["image"] = resize(data["image"].flatten(0, 1), IMAGE_SIZE).unflatten(0, data.shape)
            data["image"] = data["image"].permute(0, 1, 3, 4, 2)
            data["image"] = data["image"][..., :3] / 255.0
            data["cont"] = (1.0 - data["is_terminal"].float()).unsqueeze(-1)
            data["is_first"] = data["is_first"].float()

            post, context, metrics = model._train(data)
            
            metrics = {k: np.mean(v) for k, v in metrics.items()}
            if i % 20 == 0:
                print(f"Epoch {epoch}, iteration {i}")
                pprint(metrics)

            if (i + 1) % 500 == 0:
                with torch.no_grad():
                    results = model.video_pred(data).cpu() * 255.0
                path = os.path.join(run.dir, f"video-{epoch}-{i}.mp4")
                write_video(path, results[0], fps=10)
                metrics["video"] = wandb.Video(path, fps=10)

            run.log(metrics)
        ckpt_path = os.path.join(run.dir, f"ckpt-{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
    

if __name__ == "__main__":
    main()
    