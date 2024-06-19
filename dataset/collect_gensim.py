"""Data collection script."""

import os
import hydra
import numpy as np
import torch
import cliport
import termcolor

from cliport import tasks
from cliport.environments.environment import Environment as _Environment
import random
import logging
import time
import pybullet as p
import imageio
import einops
from collections import defaultdict

def dict_stack(dicts):
    keys = dicts[0].keys()
    result = {}
    for k in keys:
        result[k] = np.stack([d[k] for d in dicts])
    return result

class Environment(_Environment):
    
    decimation: int = 8 # 240 / 8 = 30Hz

    def __init__(self, assets_root, task=None, disp=False, shared_memory=False, hz=240, record_cfg=None):
        super().__init__(assets_root, task, disp, shared_memory, hz, record_cfg)
        # self.agent_cams[0]["image_size"] = (180, 240)
        # self.agent_cams[0]["intrinsics"] = (225, *self.agent_cams[0]["intrinsics"][1:])
        self.agent_cams[0]["image_size"] = (180, 240)
        self.agent_cams[0]["intrinsics"] = (160, *self.agent_cams[0]["intrinsics"][1:])

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.step_counter = 0
        self._traj = defaultdict(list)
        self._seed = seed
        obs = super().reset()
        info = self.info
        return obs, info
    
    def step(self, action=None):
        if action is not None:
            action_high = {
                "pose0": np.concatenate(action["pose0"]),
                "pose1": np.concatenate(action["pose1"]),
            }
            self._traj["action_high"].append(action_high)
            self._traj["high_step"].append(len(self._traj["action_low"]))
        else:
            # called by `.reset`
            # get init state
            jstate = p.getJointStates(self.ur5, self.joints)
            currj = np.array([state[0] for state in jstate])
            currjdot = np.array([state[1] for state in jstate])
            rgb, depth, seg = self.render_camera(self.agent_cams[0])
            low_obs = {
                "state": np.concatenate([currj, currjdot]),
                "rgb": rgb,
                "depth": einops.rearrange(depth, "h w -> h w 1")
            }
            self._traj["obs_low"].append(low_obs)

        obs, reward, done, info = super().step(action)
        self._traj["lang_goal"].append(info["lang_goal"])
        self._traj["obs_high"].append({
            "color_0": obs["color"][0],
            "color_1": obs["color"][1],
            "color_2": obs["color"][2],
            "depth_0": obs["depth"][0],
            "depth_1": obs["depth"][1],
            "depth_2": obs["depth"][2],
        })
        return obs, reward, done, info
    
    def movej(self, targj, speed=0.01, timeout=150):
        """Move UR5 to target joint configuration."""

        t0 = time.time()
        while (time.time() - t0) < timeout:
            # currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
            # currj = np.array(currj)
            jstate = p.getJointStates(self.ur5, self.joints)
            currj = np.array([state[0] for state in jstate])
            currjdot = np.array([state[1] for state in jstate])

            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            
            if (self.step_counter+1) % self.decimation == 0:
                rgb, depth, seg = self.render_camera(self.agent_cams[0])
                self._traj["obs_low"].append({
                    "state": np.concatenate([currj, currjdot]),
                    "rgb": rgb,
                    "depth": einops.rearrange(depth, "h w -> h w 1")
                })
                self._traj["action_low"].append(stepj)

            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            self.step_simulation()
        
        print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
        return True
    
    def get_episode_data(self, rgb_array: bool=False):
        episode_len_high = len(self._traj["action_high"])
        episode_len_low = len(self._traj["obs_low"])

        is_terminal = np.zeros(episode_len_low, dtype=bool)
        is_first    = np.zeros(episode_len_low, dtype=bool)
        is_terminal[-1] = True
        is_first[0] = True
        
        obs_high = dict_stack(self._traj["obs_high"])
        obs_high = {k: v[:-1] for k, v in obs_high.items()} # exclude last obs
        action_high = dict_stack(self._traj["action_high"])
        assert len(obs_high["color_0"]) == len(action_high["pose0"]) == episode_len_high
        
        obs_low = dict_stack(self._traj["obs_low"])
        action_low = np.stack(self._traj["action_low"])
        assert len(obs_low["rgb"]) == len(action_low) == episode_len_low

        episode_data = {
            "obs_high": obs_high,
            "action_high": action_high,
            "obs_low": obs_low,
            "action_low": action_low,
            "high_step": self._traj["high_step"],
            "is_first": is_first,
            "is_terminal": is_terminal,
            "episode_len_high": episode_len_high,
            "episode_len_low": episode_len_low,
            "seed": self._seed,
        }

        if rgb_array:
            return episode_data, obs_high["color_0"]
        else:
            return episode_data

GENSIM_ROOT = os.environ.get("GENSIM_ROOT")

@hydra.main(config_path=f'{GENSIM_ROOT}/cliport/cfg', config_name='data')
def main(cfg):
    # Initialize environment and task.
    logging.info(f"Task: {cfg['task']}, mode: {cfg['mode']}, record: {cfg['record']}")

    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=240,
        record_cfg=cfg['record']
    )
    cfg['task'] = cfg['task'].replace("_", "-")
    task: tasks.Task = tasks.names[cfg['task']]()
    task.mode = cfg['mode']

    agent = task.oracle(env)
    dataset_path = os.path.join(GENSIM_ROOT, "data", cfg["mode"], cfg["task"])
    os.makedirs(dataset_path, exist_ok=True)
    
    meta_path = os.path.join(dataset_path, "meta.pt")
    if not os.path.exists(meta_path):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        task_desc = f"Task {task}: {task.__doc__}"
        embedding = model.encode(task_desc)
        meta_dict = {}
        meta_dict["embedding"] = embedding
        torch.save(meta_dict, meta_path)

    class SingleTaskDataset:
        def __init__(self, root_path: str) -> None:
            self.files = []
            self.root_path = root_path
            for filename in os.listdir(root_path):
                if filename.startswith("episode") and filename.endswith(".pt"):
                    path = os.path.join(self.root_path, filename)
                    self.files.append(path)

        @property
        def n_episodes(self):
            return len(self.files)

        def clear(self):
            while len(self.files):
                path = self.files.pop()
                os.remove(path)

        def add_episode(self, episode_data: dict):
            len_high = episode_data["episode_len_high"]
            len_low = episode_data["episode_len_low"]
            episode_path = os.path.join(
                dataset_path, f"episode_{self.n_episodes:03}_{len_high}_{len_low}.pt")
            msg = termcolor.colored(f"Save episode data to {episode_path}", "green")
            logging.info(msg)
            torch.save(episode_data, episode_path)
            self.files.append(episode_path)

    dataset = SingleTaskDataset(root_path=dataset_path)
    logging.info(f"There are {dataset.n_episodes} episodes existing.")
    
    if 'regenerate_data' in cfg:
        dataset.n_episodes = 0

    for seed in range(dataset.n_episodes, cfg["n"]):
        logging.info(f"Collecting episode {dataset.n_episodes}:")
        env.set_task(task)
        obs, info = env.reset(seed)
        
        total_reward = 0
        for _ in range(task.max_steps):
            act = agent.act(obs, info)
            lang_goal = info['lang_goal']
            obs, reward, done, info = env.step(act)
            total_reward += reward
            print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            if done:
                break
            
            imageio.imsave(f"{_:03}.jpg", obs["color"][0])
        episode_data, ims = env.get_episode_data(rgb_array=True)
        gif_path = os.path.join(dataset_path, f"episode_{dataset.n_episodes:03}.gif")
        imageio.mimsave(gif_path, ims, format="gif")
        dataset.add_episode(episode_data)
    

if __name__ == '__main__':
    main()
