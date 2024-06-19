import torch
import os
import logging
import json
import inspect
import numpy as np

from tqdm.auto import tqdm
from torch.utils.data import Dataset
from tensordict import TensorDict, MemoryMappedTensor
from torchvision.transforms.functional import resize
from collections import defaultdict
from typing import Mapping

class GensimDataset(Dataset):

    image_size = (180, 240)
    
    def __init__(self, data: TensorDict, seq_length: int, high_level: bool=False):
        super().__init__()
        self.data = data
        self.total_length = data.shape[0]
        self.seq_length = seq_length
        self.high_level = high_level

        self.starts = self.data["is_first"].nonzero().squeeze(-1)
        self.ends = self.data["is_terminal"].nonzero().squeeze(-1)
        assert len(self.starts) == len(self.ends)
        
        try:
            from cliport.tasks import names
            self.task_desc = {}
            for task_name in self.tasks:
                task_cls = names[task_name]
                self.task_desc[task_name] = f"Task {task_name}: " + inspect.getdoc(task_cls)
        except:
            pass
    
    def __len__(self):
        return self.total_length - self.seq_length + 1

    def __getitem__(self, index):
        if isinstance(index, int):
            data = self.data[index: index+self.seq_length]
            return data
        else:
            data = torch.stack([self[i] for i in index])
            return data

    @classmethod
    def make(cls, root_path, seq_length=20, tasks: list[str]=None, high_level: bool=False):
        
        if tasks is None:
            tasks = os.listdir(root_path)

        file_paths = []
        episode_lengths_high = []
        episode_lengths_low = []
        for task in tasks:
            task_path = os.path.join(root_path, task)
            file_names = sorted([
                filename for filename in os.listdir(task_path) 
                if (filename.startswith("episode") and filename.endswith(".pt"))
            ])

            for file_name in file_names:
                file_paths.append(os.path.join(task_path, file_name))
                split = file_name[:-3].split("_")
                episode_len_high = int(split[-2])
                episode_len_low = int(split[-1])
                episode_lengths_high.append(episode_len_high)
                episode_lengths_low.append(episode_len_low)
        
        total_length_high = sum(episode_lengths_high)
        total_length_low = sum(episode_lengths_low)
        
        if high_level:
            data = cls._make_high(file_paths, total_length_high, memmap_path=os.path.join(root_path, "memmaped_high"))
        else:
            data = cls._make_low(file_paths, total_length_low, memmap_path=os.path.join(root_path, "memmaped_low"))

        return cls(data, seq_length, high_level)
    
    @staticmethod
    def _make_low(file_paths, total_length: int):

        return
    
    @staticmethod
    def _make_high(file_paths, total_length: int, memmap_path):
        print(f"Loading {total_length} high steps from {len(file_paths)} episodes")

        data = TensorDict({
            "image": MemoryMappedTensor.empty(total_length, 3, *GensimDataset.image_size),
            "state": MemoryMappedTensor.empty(total_length, 12),
            "action": MemoryMappedTensor.empty(total_length, 14),
            "episode_id": MemoryMappedTensor.empty(total_length, dtype=torch.int),
            "is_first": MemoryMappedTensor.empty(total_length, dtype=torch.bool),
            "is_terminal": MemoryMappedTensor.empty(total_length, dtype=torch.bool),
        }, [total_length])

        data.memmap_(memmap_path)

        cursor = 0
        for i, file_path in enumerate(tqdm(file_paths)):
            episode_data = torch.load(file_path)
            image = torch.as_tensor(episode_data["obs_high"]["color_0"])
            state = torch.as_tensor(episode_data["obs_low"]["state"])
            state = state[episode_data["high_step"]]
            
            action = episode_data["action_high"]
            action = np.concatenate([action["pose0"], action["pose1"]], axis=1)
            
            episode_len = len(image)
            episode_data = TensorDict({
                "image": image.permute(0, 3, 1, 2),
                "state": state,
                "action": torch.as_tensor(action),
                "episode_id": torch.full((episode_len,), i, dtype=torch.int64),
                "is_first": torch.zeros(episode_len, dtype=torch.bool),
                "is_terminal": torch.zeros(episode_len, dtype=torch.bool),
            }, [episode_len])
            episode_data["is_first"][0] = True
            episode_data["is_terminal"][-1] = True

            data[cursor: cursor+episode_len] = episode_data
            del episode_data
            cursor += episode_len
        
        # if not cursor == total_length:
        #     raise ValueError(f"Expected {total_length} but got {cursor}")

        return data


    @classmethod
    def load(cls, root_path, seq_length=20, high_level: bool=False):
        if high_level:
            data = TensorDict.load_memmap(os.path.join(root_path, "memmaped_high"))
        else:
            data = TensorDict.load_memmap(os.path.join(root_path, "memmaped_low"))
        return cls(data, seq_length, high_level)

if __name__ == "__main__":
    GensimDataset.make("/home/btx0424/gensim_ws/GenSim/data/train", high_level=True)
    GensimDataset.load("/home/btx0424/gensim_ws/GenSim/data/train", high_level=True)
