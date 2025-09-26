#!/usr/bin/env python3
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
import tqdm
import json
import random
random.seed(None)
import time
from agentlace.inference import InferenceClient
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent

import sys
sys.path.insert(0, str(root_path))

from pi0.src.openpi.training.rl_cfg import rl_config, RoboTwinEnv

config = rl_config()
seed_file = root_path / "seed_174458.json"


def print_green(x):
    print("\033[92m {}\033[00m".format(x))
def print_red(x):
    print("\033[91m {}\033[00m".format(x))
def print_yellow(x):
    print("\033[93m {}\033[00m".format(x))
def print_blue(x):
    print("\033[94m {}\033[00m".format(x))



def actor():

    Iclient = InferenceClient('10.91.1.35', 6379)
    env_id = Iclient.call("reset", {})
    while env_id is None:
        time.sleep(2)
        env_id = Iclient.call("reset", {})

    env = RoboTwinEnv(root_path=root_path)
    with open(seed_file, "r", encoding="utf-8") as f:
        task_list = json.load(f)

    pbar = tqdm.tqdm(range(0, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        sample = random.choice(task_list)
        (obs, _, _), done = env.reset(task_name=sample['task_name'], now_seed=sample['now_seed'], execute=True, instruction=sample['instruction']), False

        while not done:
            actions = Iclient.call("prediction", obs)
            while actions is None:
                time.sleep(2)
                actions = Iclient.call("prediction", obs)

            next_obs, reward, done, info = env.step(actions)
            # info['id'] = env_id
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                dones=done,
            )
            obs = next_obs

            response = Iclient.call("record", (transition, info))
            while response is None:
                time.sleep(2)
                response = Iclient.call("record", (transition, info))

        env.close_env(env.task)




def main():
    print_green("starting actor loop")
    actor()

if __name__ == "__main__":
    main()
