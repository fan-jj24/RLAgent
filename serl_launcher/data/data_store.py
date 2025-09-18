from threading import Lock
from typing import Union, Iterable

import gymnasium as gym
import jax
from serl_launcher.data.replay_buffer import ReplayBuffer, SimpleReplayBuffer

from agentlace.data.data_store import DataStoreBase


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


class SimpleReplayBufferDataStore(SimpleReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        **kwargs,
    ):
        SimpleReplayBuffer.__init__(
            self, observation_space, action_space, capacity, **kwargs
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    def clear(self):
        with self._lock:
            super().clear()

    def insert(self, *args, **kwargs):
        with self._lock:
            return super().insert(*args, **kwargs)
        
    def sample(self, *args, **kwargs):
        with self._lock:
            return super().sample(*args, **kwargs)

    def sample_sequential(self, *args, **kwargs):
        with self._lock:
            return super().sample_sequential(*args, **kwargs)


    def latest_data_id(self) -> int:
        raise NotImplementedError

    