import torch
import os
import logging
import json
import inspect

from tqdm import tqdm
from torch.utils.data import Dataset
from tensordict import TensorDict, MemoryMappedTensor
from collections import defaultdict
from typing import Mapping


class GensimDataset(Dataset):

    def __init__(self, data: TensorDict, task_episodes: Mapping, seq_length: int):
        super().__init__()
        self.data = data
        self.task_episodes = task_episodes
        self.tasks = list(task_episodes.keys())
        self.total_length = data.shape[0]
        self.seq_length = seq_length
        self.starts = self.data["is_first"].nonzero().squeeze(-1)
        self.ends = self.data["is_terminal"].nonzero().squeeze(-1)
        assert len(self.starts) == len(self.ends)
        
        from cliport.tasks import names
        self.task_desc = {}
        for task_name in self.tasks:
            task_cls = names[task_name]
            self.task_desc[task_name] = f"Task {task_name}: " + inspect.getdoc(task_cls)
    
    def __len__(self):
        return self.total_length - self.seq_length
    
    def __getitem__(self, index):
        return self.data[index: index+self.seq_length]
    
    def episodes(self, task: str):
        starts = self.starts[self.task_episodes[task]]
        ends = self.ends[self.task_episodes[task]]
        for start, end in zip(starts, ends):
            yield self.data[start: end]

    @staticmethod
    def _parse_dir(root_path, tasks):
        if tasks is None:
            tasks = os.listdir(root_path)
            if "memmaped" in tasks:
                tasks.remove("memmaped")

        file_paths = []
        episode_lengths = []
        task_episodes = defaultdict(list)
        for task in tasks:
            task_path = os.path.join(root_path, task)
            file_names = [filename for filename in os.listdir(task_path) if filename.endswith(".pt")]
            for file_name in file_names:
                task_episodes[task].append(len(file_paths))

                file_paths.append(os.path.join(task_path, file_name))
                episode_lengths.append(int(file_name[:-3].split("_")[-1]))
        
        return file_paths, episode_lengths, task_episodes


    @classmethod
    def make(cls, root_path, seq_length=20, tasks: list[str]=None):
        
        file_paths, episode_lengths, task_episodes = GensimDataset._parse_dir(root_path, tasks)

        total_length = sum(episode_lengths)
        
        logging.info(f"Loading {total_length} steps from {len(task_episodes)} tasks")

        data = TensorDict({
            "action": MemoryMappedTensor.empty(total_length, 6),
            "state": MemoryMappedTensor.empty(total_length, 12),
            "image": MemoryMappedTensor.empty(total_length, 240, 320, 4),
            "is_first": MemoryMappedTensor.empty(total_length),
            "is_terminal": MemoryMappedTensor.empty(total_length),
        }, [total_length])

        data.memmap_(os.path.join(root_path, "memmaped"))
        with open(os.path.join(root_path, "task_episodes"), "w") as f:
            json.dump(task_episodes, f)
        
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
        return cls(data, task_episodes, seq_length)
    
    @classmethod
    def load(cls, root_path, seq_length=20):
        data = TensorDict.load_memmap(os.path.join(root_path, "memmaped"))
        task_episodes_path = os.path.join(root_path, "task_episodes")
        with open(task_episodes_path, "r") as f:
            task_episodes = json.load(f)
        return cls(data, task_episodes, seq_length)