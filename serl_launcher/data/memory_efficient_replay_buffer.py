import copy
from typing import Iterable, Optional, Tuple, Dict, Any

import gymnasium as gym
import numpy as np
from serl_launcher.data.dataset import DatasetDict, _sample
from serl_launcher.data.replay_buffer import ReplayBuffer, _insert_recursively, _init_replay_dict
from flax.core import frozen_dict
from gymnasium.spaces import Box


class MemoryEfficientReplayBuffer(ReplayBuffer):
    ### for chunk observation
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        include_next_actions: Optional[bool] = False,
        include_grasp_penalty: Optional[bool] = False,
    ):
        self.pixel_keys = pixel_keys

        observation_space = copy.deepcopy(observation_space)
        self._num_stack = None
        for pixel_key in self.pixel_keys:
            pixel_obs_space = observation_space.spaces[pixel_key]
            if self._num_stack is None:
                self._num_stack = pixel_obs_space.shape[0]
            else:
                assert self._num_stack == pixel_obs_space.shape[0]
            self._unstacked_dim_size = pixel_obs_space.shape[-1]
            low = pixel_obs_space.low[0]
            high = pixel_obs_space.high[0]
            unstacked_pixel_obs_space = Box(
                low=low, high=high, dtype=pixel_obs_space.dtype
            )
            observation_space.spaces[pixel_key] = unstacked_pixel_obs_space

        next_observation_space_dict = copy.deepcopy(observation_space.spaces)
        for pixel_key in self.pixel_keys:
            next_observation_space_dict.pop(pixel_key)
        next_observation_space = gym.spaces.Dict(next_observation_space_dict)

        self._first = True
        self._is_correct_index = np.full(capacity, False, dtype=bool)

        super().__init__(
            observation_space,
            action_space,
            capacity,
            next_observation_space=next_observation_space,
            include_next_actions=include_next_actions,
            include_grasp_penalty=include_grasp_penalty,
        )

    def insert(self, data_dict: DatasetDict):
        if self._insert_index == 0 and self._capacity == len(self) and not self._first:
            indxs = np.arange(len(self) - self._num_stack, len(self))
            for indx in indxs:
                element = super().sample(1, indx=indx)
                self._is_correct_index[self._insert_index] = False
                super().insert(element)

        data_dict = data_dict.copy()
        data_dict["observations"] = data_dict["observations"].copy()
        data_dict["next_observations"] = data_dict["next_observations"].copy()

        obs_pixels = {}
        next_obs_pixels = {}
        for pixel_key in self.pixel_keys:
            obs_pixels[pixel_key] = data_dict["observations"].pop(pixel_key)
            next_obs_pixels[pixel_key] = data_dict["next_observations"].pop(pixel_key)

        if self._first:
            for i in range(self._num_stack):
                for pixel_key in self.pixel_keys:
                    data_dict["observations"][pixel_key] = obs_pixels[pixel_key][i]

                self._is_correct_index[self._insert_index] = False
                super().insert(data_dict)

        for pixel_key in self.pixel_keys:
            data_dict["observations"][pixel_key] = next_obs_pixels[pixel_key][-1]

        self._first = data_dict["dones"]

        self._is_correct_index[self._insert_index] = True
        super().insert(data_dict)

        for i in range(self._num_stack):
            indx = (self._insert_index + i) % len(self)
            self._is_correct_index[indx] = False

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        pack_obs_and_next_obs: bool = False,
    ) -> frozen_dict.FrozenDict:
        """Samples from the replay buffer.

        Args:
            batch_size: Minibatch size.
            keys: Keys to sample.
            indx: Take indices instead of sampling.
            pack_obs_and_next_obs: whether to pack img and next_img into one image.
                It's useful when they have overlapping frames.

        Returns:
            A frozen dictionary.
        """

        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

            for i in range(batch_size):
                while not self._is_correct_index[indx[i]]:
                    if hasattr(self.np_random, "integers"):
                        indx[i] = self.np_random.integers(len(self))
                    else:
                        indx[i] = self.np_random.randint(len(self))
        else:
            raise NotImplementedError()

        if keys is None:
            keys = self.dataset_dict.keys()
        else:
            assert "observations" in keys

        keys = list(keys)
        keys.remove("observations")
        batch = super().sample(batch_size, keys, indx)
        batch = batch.unfreeze()

        obs_keys = self.dataset_dict["observations"].keys()
        obs_keys = list(obs_keys)
        for pixel_key in self.pixel_keys:
            obs_keys.remove(pixel_key)

        batch["observations"] = {}
        for k in obs_keys:
            batch["observations"][k] = _sample(
                self.dataset_dict["observations"][k], indx
            )

        for pixel_key in self.pixel_keys:
            obs_pixels = self.dataset_dict["observations"][pixel_key]
            obs_pixels = np.lib.stride_tricks.sliding_window_view(
                obs_pixels, self._num_stack + 1, axis=0
            )
            obs_pixels = obs_pixels[indx - self._num_stack]
            # transpose from (B, H, W, C, T) to (B, T, H, W, C) to follow jaxrl_m convention
            obs_pixels = obs_pixels.transpose((0, 4, 1, 2, 3))

            if pack_obs_and_next_obs:
                batch["observations"][pixel_key] = obs_pixels
            else:
                batch["observations"][pixel_key] = obs_pixels[:, :-1, ...]
                if "next_observations" in keys:
                    batch["next_observations"][pixel_key] = obs_pixels[:, 1:, ...]

        return frozen_dict.freeze(batch)



class DynamicNextObsReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        # 只保留必要的字段
        dataset_dict = {
            "observations": _init_replay_dict(observation_space, capacity),
            "actions": np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            "rewards": np.empty((capacity,), dtype=np.float32),
            "dones": np.empty((capacity,), dtype=bool),
        }
        self.observation_space = observation_space

        super().__init__(observation_space, action_space, capacity)
        self.dataset_dict = dataset_dict
        self._last_next_obs = None  # 保存最新传入的 next_obs
        self._latest_index = 0      # 最新插入的索引
        self._size = 0              # 当前 buffer 中的样本数

    def insert(self, data_dict: Dict[str, Any]):
        """
        插入数据时,data_dict 必须包含 'next_observations' 字段。
        但 buffer 不存储 next_observations,只记录当前的 obs, action, reward, done。
        """
        # 保存当前样本的 next_observations 到 last_next_obs
        self._last_next_obs = data_dict.get("next_observations", None)

        # 构造仅包含当前样本的 obs, action, reward, done 的 data_dict
        reduced_data_dict = {
            "observations": data_dict["observations"],
            "actions": data_dict["actions"],
            "rewards": data_dict["rewards"],
            "dones": data_dict["dones"],
        }

        # 插入到 buffer 中
        for k in reduced_data_dict:
            if isinstance(self.dataset_dict[k], dict):
                _insert_recursively(self.dataset_dict[k], reduced_data_dict[k], self._latest_index)
            else:
                self.dataset_dict[k][self._latest_index] = reduced_data_dict[k]

        self._latest_index = (self._latest_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> frozen_dict.FrozenDict:
        """
        采样时动态构建 next_observations 字段。
        """
        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

        if keys is None:
            keys = list(self.dataset_dict.keys())
        else:
            keys = list(keys)

        # 从原始数据中采样
        batch = {}
        batch["next_actions"] = self.dataset_dict["actions"][(indx + 1) % self._capacity]

        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx)
            else:
                batch[k] = self.dataset_dict[k][indx]

        # 动态构建 next_observations 的嵌套字典结构
        next_observations = _init_replay_dict(self.observation_space, batch_size)

        # 填充 next_observations 数据（递归）
        def fill_next_obs(next_obs_dict, obs_dict, idx, next_idx, done):
            for k in next_obs_dict:
                if isinstance(obs_dict[k], dict):
                    fill_next_obs(next_obs_dict[k], obs_dict[k], idx, next_idx, done)
                else:
                    if done:
                        next_obs_dict[k][idx] = np.zeros_like(next_obs_dict[k][idx])
                    else:
                        next_obs_dict[k][idx] = obs_dict[k][next_idx]

        
        for i in range(batch_size):
            current_indx = indx[i]
            current_done = batch["dones"][i]

            if current_done:
                # done=True: 填零
                fill_next_obs(next_observations, self.dataset_dict["observations"], i, 0, True)
            elif current_indx == (self._latest_index - 1 + self._capacity) % self._capacity:
                # 最新样本，用 last_next_obs
                _insert_recursively(next_observations, self._last_next_obs, i)
            else:
                # 否则取下一个 obs
                next_indx = (current_indx + 1) % self._capacity
                fill_next_obs(next_observations, self.dataset_dict["observations"], i, next_indx, False)

        batch["next_observations"] = next_observations
        return frozen_dict.freeze(batch)
    

class SimpleReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        dataset_dict = {
            "observations": _init_replay_dict(observation_space, capacity),
            "actions": np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            "rewards": np.empty((capacity,), dtype=np.float32),
            "dones": np.empty((capacity,), dtype=bool),
        }
        self.observation_space = observation_space

        super().__init__(observation_space, action_space, capacity)
        self.dataset_dict = dataset_dict
        self._latest_index = 0
        self._size = 0

    def insert(self, data_dict: Dict[str, Any]):

        reduced_data_dict = {
            "observations": data_dict["observations"],
            "actions": data_dict["actions"],
            "rewards": data_dict["rewards"],
            "dones": data_dict["dones"],
        }

        for k in reduced_data_dict:
            if isinstance(self.dataset_dict[k], dict):
                _insert_recursively(self.dataset_dict[k], reduced_data_dict[k], self._latest_index)
            else:
                self.dataset_dict[k][self._latest_index] = reduced_data_dict[k]

        self._latest_index = (self._latest_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> frozen_dict.FrozenDict:

        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

        if keys is None:
            keys = list(self.dataset_dict.keys())
        else:
            keys = list(keys)

        batch = {}
        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx)
            else:
                batch[k] = self.dataset_dict[k][indx]

        return frozen_dict.freeze(batch)