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

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

DATASET_ROOT = "/media/aa/hdd1/cf/data/train"
IMAGE_SIZE = (128, 128)


class RoboGenDataset(Dataset):
    def __init__(self, data: TensorDict, seq_length: int):
        super().__init__()
        self.data = data
        self.total_length = data.shape[0]
        self.seq_length = seq_length
        
        self.starts = self.data["is_first"].nonzero().squeeze(-1)
        self.ends = self.data["is_terminal"].nonzero().squeeze(-1)

    def __len__(self):
        return self.total_length - self.seq_length + 1
    
    def __getitem__(self, index):
        if isinstance(index, int):
            data = self.data[index: index+self.seq_length]
        else:
            data = torch.stack([self[i] for i in index])
        return data
    
    @classmethod
    def make(cls, root_dir: str, seq_length: int):
        task_episodes = {}
        task_lengths = {}
        for task in os.listdir(root_dir):
            task_path = os.path.join(root_dir, task)
            episode_paths = [
                os.path.join(task_path, episode) 
                for episode in os.listdir(task_path)
                if episode.endswith(".pt")
            ]
            episode_lengths = [
                int(episode[:-3].split("_")[-1])
                for episode in os.listdir(task_path)
                if episode.endswith(".pt")
            ]
            task_episodes[task] = episode_paths
            task_lengths[task] = sum(episode_lengths)
        total_length = sum(task_lengths.values())
        print(f"Total length: {total_length}")

        data = TensorDict({
            "image": MemoryMappedTensor.empty(total_length, *IMAGE_SIZE, 3),
            "action": MemoryMappedTensor.empty(total_length, 7),
            "is_first": MemoryMappedTensor.empty(total_length),
            "is_terminal": MemoryMappedTensor.empty(total_length),
        }, [total_length])

        cursor = 0
        for task, episode_paths in tqdm(task_episodes.items()):
            print(f"{task}: {len(episode_paths)} episodes, {task_lengths[task]} steps")
            for episode_path in episode_paths:
                episode_data = torch.load(episode_path)
                image = torch.as_tensor(episode_data["image"], dtype=torch.float32)
                state = torch.as_tensor(episode_data["state"], dtype=torch.float32)
                action = torch.as_tensor(episode_data["action"], dtype=torch.float32)
                is_first = torch.as_tensor(episode_data["is_first"], dtype=bool)
                is_terminal = torch.as_tensor(episode_data["is_terminal"], dtype=bool)
                length = episode_data["length"]

                image = image.permute(0, 3, 1, 2)
                image = resize(image, IMAGE_SIZE)
                image = image.permute(0, 2, 3, 1)

                episode_data = TensorDict({
                    "image": image,
                    "state": state[..., -7:],
                    "action": action,
                    "is_first": is_first,
                    "is_terminal": is_terminal
                }, [length])

                data[cursor: cursor + length] = episode_data
                cursor += length
        assert cursor == total_length
        return cls(data, seq_length)


dataset = RoboGenDataset.make(DATASET_ROOT, 40)


@hydra.main(config_path=os.path.join(FILE_PATH, "cfg"), config_name="train")
def main(cfg):
    device = cfg.world_model.device

    run = wandb.init(project="GensimEval", entity="btx0424")
    run.config["project"] = "robogen"

    model = WorldModel(
        obs_space=gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, (*IMAGE_SIZE, 3), dtype=np.uint8),
                "state": gym.spaces.Box(0, 20, (7,), dtype=np.float32),
            }
        ),
        act_space=gym.spaces.Box(-1, 1, (7,), dtype=np.float32),
        step=0,
        config=cfg.world_model,
    ).to(device)

    if cfg.get("ckpt", None) is not None:
        print(f"load checkpoint from {cfg.ckpt}")
        ckpt = torch.load(cfg.ckpt)
        model.load_state_dict(ckpt)
        del ckpt

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        collate_fn=torch.stack
    )

    for epoch in range(cfg.get("epochs", 10)):
        for i, data in enumerate(dataloader):
            # some manual processing
            data = data.to(device)
            data["image"] = data["image"][..., :3] / 255.0
            data["cont"] = (1.0 - data["is_terminal"]).unsqueeze(-1)

            post, context, metrics = model._train(data)
            
            metrics = {k: np.mean(v) for k, v in metrics.items()}
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
    