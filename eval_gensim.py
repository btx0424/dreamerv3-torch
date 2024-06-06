import torch
import numpy as np
import gym
import gym.spaces
import os
import hydra

from torch.utils.data import DataLoader
from torchvision.io import write_video
from torchvision.utils import make_grid
from pprint import pprint

from models import WorldModel
from dataset.gensim import GensimDataset

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

@hydra.main(config_path=FILE_PATH, config_name="train")
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

    data_dir = "/home/btx0424/gensim_ws/GenSim/data"
    # data_dir = os.environ.get("GENSIM_PATH")
    try:
        dataset = GensimDataset.load(data_dir)
    except:
        dataset = GensimDataset.make(data_dir)
    

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        collate_fn=torch.stack
    )

    for epoch in range(cfg.get("epochs", 5)):
        for i, data in enumerate(dataloader):
            # some manual processing
            data = data.to(device)
            data["image"] = data["image"][..., :3] / 255.0
            data["cont"] = (1.0 - data["is_terminal"]).unsqueeze(-1)

            post, context, metrics = model._train(data)
            print(f"iteration: {i}, model loss: {metrics["model_loss"].item()}")
            
            if (i + 1) % 200 == 0:
                with torch.no_grad():
                    results = model.video_pred(data).cpu() * 255.0
                write_video(f"video-{i}.mp4", results[0], fps=10)

    

if __name__ == "__main__":
    main()
    