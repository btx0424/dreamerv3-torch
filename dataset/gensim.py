import torch
import os
import logging
import json
import inspect

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
        
        if self.high_level:
            if self.seq_length > 2:
                raise ValueError
            self.high_action_idx = data["high_action_mask"].nonzero().squeeze(-1)
            self.total_length = len(self.high_action_idx)
    
    def __len__(self):
        return self.total_length - self.seq_length + 1

    def __getitem__(self, index):
        if isinstance(index, int):
            if self.high_level:
                index = self.high_action_idx[index: index+self.seq_length]
                data = self.data[index]
                data["action"] = data["high_action"]
            else:
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
        episode_lengths = []
        for task in tasks:
            task_path = os.path.join(root_path, task)
            file_names = sorted([
                filename for filename in os.listdir(task_path) 
                if (filename.startswith("episode") and filename.endswith(".pt"))
            ])
            print(file_names[:5])
            for file_name in file_names[:5]:
                file_paths.append(os.path.join(task_path, file_name))
                episode_lengths.append(int(file_name[:-3].split("_")[-1]))
        
        total_length = sum(episode_lengths)
        
        logging.info(f"Loading {total_length} steps from {len(tasks)} tasks")

        data = TensorDict({
            "high_action_mask": MemoryMappedTensor.empty(total_length, dtype=bool),
            "high_action": MemoryMappedTensor.empty(total_length, 14),
            "high_image": MemoryMappedTensor.empty(total_length, *GensimDataset.image_size, 4),
            "action": MemoryMappedTensor.empty(total_length, 6),
            "state": MemoryMappedTensor.empty(total_length, 12),
            "image": MemoryMappedTensor.empty(total_length, *GensimDataset.image_size, 4),
            "is_first": MemoryMappedTensor.empty(total_length, dtype=bool),
            "is_terminal": MemoryMappedTensor.empty(total_length, dtype=bool),
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

            if episode_data.shape[1:3] != GensimDataset.image_size:
                episode_data["image"] = episode_data["image"].permute(0, 3, 1, 2)
                episode_data["image"] = resize(episode_data["image"], GensimDataset.image_size)
                episode_data["image"] = episode_data["image"].permute(0, 2, 3, 1)
            
            if episode_data.get("high_action_mask", None) is None:
                episode_data["high_action"] = torch.zeros(episode_len, 14)
                episode_data["high_action_mask"] = torch.zeros(episode_len, dtype=bool)

            data[ i: i+episode_len] = episode_data
            del episode_data
            i += episode_len
            t.update(episode_len)
        return cls(data, seq_length, high_level)
    
    @classmethod
    def load(cls, root_path, seq_length=20, high_level: bool=False):
        data = TensorDict.load_memmap(os.path.join(root_path, "memmaped"))
        return cls(data, seq_length, high_level)

if __name__ == "__main__":
    GensimDataset.make("/home/btx0424/gensim_ws/GenSim/data")