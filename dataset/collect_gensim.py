"""Data collection script."""

import os
import hydra
import numpy as np
import torch
import cliport

from cliport import tasks
from cliport.environments.environment import Environment as _Environment
import random
import logging
import time
import pybullet as p
import imageio
import einops
from collections import defaultdict

class Environment(_Environment):
    
    decimation: int = 8 # 240 / 8 = 30Hz

    def __init__(self, assets_root, task=None, disp=False, shared_memory=False, hz=240, record_cfg=None):
        super().__init__(assets_root, task, disp, shared_memory, hz, record_cfg)
        # self.agent_cams[0]["image_size"] = (240, 320)
        # self.agent_cams[0]["intrinsics"] = (225, *self.agent_cams[0]["intrinsics"][1:])
        self.agent_cams[0]["image_size"] = (180, 240)
        self.agent_cams[0]["intrinsics"] = (160, *self.agent_cams[0]["intrinsics"][1:])

    def reset(self):
        self._joint_action = []
        self._high_action = defaultdict(list)
        # self._high_obs = []
        self.step_counter = 0
        self._obs = defaultdict(list)
        obs = super().reset()
        return obs
    
    def step(self, action=None):
        if action is not None:
            for k, v in action.items():
                self._high_action[k].append(np.concatenate(v))
            self._high_action["t"].append(len(self._joint_action))
        obs, reward, done, info = super().step(action)
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
            
            if self.step_counter % self.decimation == 0:
                self._joint_action.append(stepj)
                self._obs["state"].append(np.concatenate([currj, currjdot]))
                rgb, depth, seg = self.render_camera(self.agent_cams[0])
                self._obs["rgb"].append(rgb)
                self._obs["depth"].append(einops.rearrange(depth, "h w -> h w 1"))

            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            self.step_counter += 1
            self.step_simulation()
        
        print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
        return True
    
    def get_episode_data(self, rgb_array: bool=False):
        episode_len = len(self._joint_action)
        is_terminal = np.zeros(episode_len, dtype=bool)
        is_first = np.zeros(episode_len, dtype=bool)
        is_terminal[-1] = True
        is_first[0] = True
        rgb = np.stack(self._obs["rgb"])
        depth = np.stack(self._obs["depth"])

        action = np.concatenate([
            np.stack(self._high_action["pose0"]),
            np.stack(self._high_action["pose1"])
        ], axis=-1)
        high_action = np.zeros((episode_len, 14))
        high_action[self._high_action["t"]] = action
        high_action_mask = np.zeros((episode_len,), dtype=bool)
        high_action_mask[self._high_action["t"]] = True
        
        episode_data = {
            "high_action": high_action,
            "high_action_mask": high_action_mask,
            "action": np.stack(self._joint_action),
            "image": np.concatenate([rgb, depth], axis=-1),
            "state": np.stack(self._obs["state"]),
            "is_first": is_first,
            "is_terminal": is_terminal,
            "length": episode_len
        }
        if rgb_array:
            return episode_data, rgb
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
    dataset_path = os.path.join(GENSIM_ROOT, "data", cfg["task"])
    os.makedirs(dataset_path, exist_ok=True)

    class SingleTaskDataset:
        def __init__(self, root_path: str) -> None:
            self.files = []
            self.root_path = root_path
            for filename in os.listdir(root_path):
                if filename.endswith(".pt"):
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
            episode_path = os.path.join(dataset_path, f"episode_{self.n_episodes}_{episode_data['length']}.pt")
            logging.info(f"Save episode data to {episode_path}")
            torch.save(episode_data, episode_path)
            self.files.append(episode_path)

    dataset = SingleTaskDataset(root_path=dataset_path)
    logging.info(f"There are {dataset.n_episodes} episodes existing.")
    
    if 'regenerate_data' in cfg:
        dataset.n_episodes = 0

    while dataset.n_episodes < cfg["n"]:
        logging.info(f"Collecting episode {dataset.n_episodes}:")
        env.set_task(task)
        obs = env.reset()
        info = env.info
        
        total_reward = 0
        for _ in range(task.max_steps):
            act = agent.act(obs, info)
            lang_goal = info['lang_goal']
            obs, reward, done, info = env.step(act)
            total_reward += reward
            print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            if done:
                break
    
        episode_data, ims = env.get_episode_data(rgb_array=True)
        imageio.mimsave(os.path.join(dataset_path, f"episode_{dataset.n_episodes}.gif"), ims, format="gif")
        dataset.add_episode(episode_data)
    

if __name__ == '__main__':
    main()
