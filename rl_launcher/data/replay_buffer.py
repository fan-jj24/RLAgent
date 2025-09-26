import collections
from typing import Any, Iterator, Optional, Sequence, Tuple, Union, Dict, Iterable

import gymnasium as gym
import jax
import numpy as np
from serl_launcher.data.dataset import Dataset, DatasetDict, _sample


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        include_next_actions: Optional[bool] = False,
        include_label: Optional[bool] = False,
        include_grasp_penalty: Optional[bool] = False,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        if include_next_actions:
            dataset_dict['next_actions'] = np.empty((capacity, *action_space.shape), dtype=action_space.dtype)
            dataset_dict['next_intvn'] = np.empty((capacity,), dtype=bool)
            
        if include_label:
            dataset_dict['labels'] = np.empty((capacity,), dtype=int)
        
        if include_grasp_penalty:
            dataset_dict['grasp_penalty'] = np.empty((capacity,), dtype=np.float32)

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}, device=None):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data, device=device))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def download(self, from_idx: int, to_idx: int):
        indices = np.arange(from_idx, to_idx)
        data_dict = self.sample(batch_size=len(indices), indx=indices)
        return to_idx, data_dict

    def get_download_iterator(self):
        last_idx = 0
        while True:
            if last_idx >= self._size:
                raise RuntimeError(f"last_idx {last_idx} >= self._size {self._size}")
            last_idx, batch = self.download(last_idx, self._size)
            yield batch

    
class SimpleReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        dataset_dict = {
            "observations": _init_replay_dict(observation_space, capacity),
            "noises": np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            "log_probs": np.empty((capacity,), dtype=np.float32),
            "advantages": np.empty((capacity,), dtype=np.float32),
            # "dones": np.empty((capacity,), dtype=np.int8),
        }
        self.observation_space = observation_space

        super().__init__(observation_space, action_space, capacity)
        self.dataset_dict = dataset_dict
        self._latest_index = 0
        self._size = 0

    def clear(self):
        def _zero_recursively(arr):
            if isinstance(arr, dict):
                for k in arr:
                    _zero_recursively(arr[k])
            else:
                arr[:] = 0

        self._size = 0
        self._latest_index = 0
        for key in self.dataset_dict:
            _zero_recursively(self.dataset_dict[key])

    def __len__(self) -> int:
        return self._size
    
    def insert(self, data_dict: Dict[str, Any]):

        reduced_data_dict = {
            "observations": data_dict["observations"],
            "noises": data_dict["noises"],
            "log_probs": data_dict["log_probs"],
            "advantages": data_dict["advantages"],
            # "dones": data_dict["dones"],
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
    ):

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

        return batch

    def sample_sequential(
        self,
        batch_size: int,
        start_idx: int,
        keys: Optional[Iterable[str]] = None,
    ):
        end_idx = min(start_idx + batch_size, self._size)
        indx = np.arange(start_idx, end_idx)

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

        return batch, end_idx - 1

