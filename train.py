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

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class GensimDataset(Dataset):

    def __init__(self, data: TensorDict, seq_length: int):
        super().__init__()
        self.data = data
        self.total_length = data.shape[0]
        self.seq_length = seq_length
    
    def __len__(self):
        return self.total_length - self.seq_length
    
    def __getitem__(self, index):
        return self.data[index: index+self.seq_length]

    @classmethod
    def make(cls, root_path, seq_length=20):
        tasks = os.listdir(root_path)
        file_paths = []
        episode_lengths = []
        for task in tasks:
            task_path = os.path.join(root_path, task)
            file_names = [filename for filename in os.listdir(task_path) if filename.endswith(".pt")]
            for file_name in file_names:
                file_paths.append(os.path.join(task_path, file_name))
                episode_lengths.append(int(file_name[:-3].split("_")[-1]))
        
        total_length = sum(episode_lengths)
        
        logging.info(f"Loading {total_length} steps from {len(tasks)} tasks")

        data = TensorDict({
            "action": MemoryMappedTensor.empty(total_length, 6),
            "state": MemoryMappedTensor.empty(total_length, 12),
            "image": MemoryMappedTensor.empty(total_length, 240, 320, 4),
            "is_first": MemoryMappedTensor.empty(total_length),
            "is_terminal": MemoryMappedTensor.empty(total_length),
        }, [total_length])

        data.memmap_(os.path.join(root_path, "memmaped"))
        
        i = 0
        t = tqdm(total=total_length)
        for file_path in file_paths:
            episode_data = torch.load(file_path)
            episode_len = episode_data.pop("length")
            episode_data = TensorDict({
                k: torch.as_tensor(v) for k, v in episode_data.items()
            }, [episode_len])
            data[ i: i+episode_len] = episode_data
            del episode_data
            i += episode_len
            t.update(episode_len)
        return cls(data, seq_length)
    
    @classmethod
    def load(cls, root_path, seq_length=20):
        data = TensorDict.load_memmap(os.path.join(root_path, "memmaped"))
        return cls(data, seq_length)


@hydra.main(config_path=FILE_PATH, config_name="train")
def main(cfg):
    device = cfg.world_model.device

    run = wandb.init(project="GensimEval", entity="btx0424")

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

    data_dir = os.environ.get("GENSIM_PATH", None)
    if data_dir is None:
        raise ValueError

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
            
            run.log(metrics)
            print(f"Epoch {epoch}, iteration {i}")
            print(OmegaConf.to_yaml(metrics))
            
            if (i + 1) % 200 == 0:
                with torch.no_grad():
                    results = model.video_pred(data).cpu() * 255.0
                write_video(f"video-{i}.mp4", results[0], fps=10)

    

if __name__ == "__main__":
    main()
    