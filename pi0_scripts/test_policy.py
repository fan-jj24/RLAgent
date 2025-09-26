#!/usr/bin/env python3
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import subprocess
import yaml

from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent

import sys
sys.path.insert(0, str(root_path))

from pi0.src.openpi.training.rl_cfg import RoboTwinEnv, rl_config
config = rl_config()

from functools import partial
from typing import Iterable, Optional, Tuple, FrozenSet
import numpy as np
import flax
import flax.linen as nn

import jax
import jax.numpy as jnp
from serl_launcher.common.common import  ModuleDict, nonpytree_field
from serl_launcher.common.state_test import JaxRLTrainState
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.utils.parmas_utils import create_policy

class Testagent(flax.struct.PyTreeNode):

    state: JaxRLTrainState
    config: dict = nonpytree_field()
    ACTOR_PARAM_KEY = 'modules_actor'

    def forward_policy(
        self,
        sample_rng,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jnp.ndarray:
        if train:
            assert rng is not None, "Must specify rng when training"
        def forward_fn(params, sample_key, obs, dropout_rng):
            actions = self.state.apply_fn(
                {"params": params},
                sample_key,
                obs,
                rngs={"dropout": dropout_rng} if train else {},
                name="actor",
                train=train,
            )
            return jax.tree.map(lambda x: x[..., :14], actions)
        return forward_fn(grad_params or self.state.params, sample_rng, observations, rng)
    

    @partial(jax.jit, static_argnames=("argmax", "action_chunk"))
    def sample_actions(
        self,
        sample_rng,
        observations: Data,
        action_chunk: int = 50,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = True,
    ) -> jnp.ndarray:
        observations = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], observations)
        if argmax:
            joint_actions = self.forward_policy(sample_rng = sample_rng, observations = observations, train=False)[0]
        else:
            assert seed is not None, "Must provide seed for sampling"
            noise = jax.random.normal(seed, shape=(action_chunk, 14)) * self.config["target_policy_noise"]
            noise = jnp.clip(noise, -self.config["noise_clip"], self.config["noise_clip"])
            joint_actions = self.forward_policy(sample_rng = sample_rng, observations = observations, train=False)[0] + noise
        return joint_actions


    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actor_def: nn.Module,
        pretrained_actor_params: Optional[Params] = None,
    ):
        networks = {"actor": actor_def}
        model_def = ModuleDict(networks)
        rng, init_rng = jax.random.split(rng)
        sample_rng, create_rng = jax.random.split(rng)
        all_params = jax.eval_shape(lambda: model_def.init(init_rng, actor=[sample_rng, observations]))["params"]


        if pretrained_actor_params is not None:
            try:
                target_actor_params = all_params[cls.ACTOR_PARAM_KEY]
                for module_name, module_pretrained_params in pretrained_actor_params.items():
                    if module_name in target_actor_params:# 直接尝试匹配顶层键
                        target_actor_params[module_name] = module_pretrained_params
                        print(f"Loaded pretrained param: {module_name}")
                    elif module_name == "PaliGemma":# 特例处理 PaliGemma，因为它包含了 img 和 llm
                        for submodule_name, submodule_pretrained_params in module_pretrained_params.items():
                            if submodule_name in target_actor_params:
                                target_actor_params[submodule_name] = submodule_pretrained_params
                                print(f"Loaded pretrained param: {submodule_name}")
                    else:
                        print(f"Warning: Pretrained module key '{module_name}' not found in initialized actor params. Skipping. Available: {list(target_actor_params.keys())}")
            except Exception as update_e:
                print(f"Error updating params with pretrained ones: {update_e}")
                raise update_e

        
        params = flax.core.freeze(all_params)

        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=None,
            rng=create_rng,
        )
        return cls(
            state=state,
            config={

            }
        )

    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations: Data,
        pretrained_policy_path: Optional[str] = None,
        lora: bool = False,
    ):

        policy_def, pretrained_actor_params = create_policy(pretrained_policy_path=pretrained_policy_path,  lora = lora)
        agent = cls.create(
            rng,
            observations,
            actor_def=policy_def,
            pretrained_actor_params=pretrained_actor_params,
        )
            
        return agent



def print_green(x):
    print("\033[92m {}\033[00m".format(x))
def print_red(x):
    print("\033[91m {}\033[00m".format(x))
def print_yellow(x):
    print("\033[93m {}\033[00m".format(x))
def print_blue(x):
    print("\033[94m {}\033[00m".format(x))


##############################################################################
def return_ffmpeg(save_dir, video_size, eval_count):
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            video_size,
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            f"{save_dir}/episode{eval_count}.mp4",
        ],
        stdin=subprocess.PIPE,
    )
    return ffmpeg



def actor():

    output_file = open(str(root_path / "test_results" / "test_results1000" / "policy_results.log"), "a")
    results_file = open(str(root_path / "test_results" / "test_results1000" / "total_results.log"), "a")
    save_dir = root_path / "test_results" / "test_results1000"
    save_dir.mkdir(parents=True, exist_ok=True)
    camera_config_path = str(root_path / "task_config" / "_camera_config.yml")
    with open(camera_config_path, "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)["D435"]
    video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])

    rng = jax.random.PRNGKey(42)
    rng, sampling_rng = jax.random.split(rng)    

    sample_obs = config.observation_space.sample()
    sample_obs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], sample_obs)
    agent = Testagent.create_pixels(
        rng=rng,
        observations=sample_obs,
        pretrained_policy_path=str(root_path / 'checkpoints' / 'params' / '1000' / 'params'))

    env = RoboTwinEnv(root_path=root_path)
    test_task_name = ["stack_blocks_two", "stack_blocks_three", "stack_bowls_two", "stack_bowls_three",]
    for i, task_name in enumerate(test_task_name):
        now_seed = 100000 - 1
        success_num, total_progress = 0, 0.0
        for eval_count in range(100):
            (obs, instruction, now_seed), done = env.reset(task_name=task_name, save_video=True, now_seed=now_seed + 1), False
            env.task.eval_video_path = save_dir
            ffmpeg = return_ffmpeg(save_dir, video_size, eval_count + i * 100)
            env.task._set_eval_video_ffmpeg(ffmpeg)
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    sample_rng=sampling_rng,
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=True,
                )
                actions = np.asarray(jax.device_get(actions))
                next_obs, reward, done, info = env.step(actions)
                obs = next_obs
                if done:
                    result_msg = f"progress: {info['progress']}, seed: {now_seed}"
                    if info['progress'] == 1.0: success_num += 1
                    total_progress += info['progress']
                    output_file.write(f"{task_name}-{eval_count}: {result_msg}\n")
                    output_file.flush()
                    env.task._del_eval_video_ffmpeg()
                    env.close_env(env.task)

        results_file.write(f"{task_name}: success_rate: {success_num / 100}, avg_progress: {total_progress / 100}\n")
        results_file.flush()


if __name__ == "__main__":
    actor()




