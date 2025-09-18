#!/usr/bin/env python3
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time
import copy
import signal
from pathlib import Path
root_path = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(root_path))
from agentlace.trainer import TrainerClient

from serl_launcher.utils.launcher import make_trainer_config
from rl_cfg import rl_config, RoboTwinEnv

config = rl_config()
env = RoboTwinEnv(root_path=root_path)
resume_flag = False
def print_green(x):
    print("\033[92m {}\033[00m".format(x))
def print_red(x):
    print("\033[91m {}\033[00m".format(x))
def print_yellow(x):
    print("\033[93m {}\033[00m".format(x))
def print_blue(x):
    print("\033[94m {}\033[00m".format(x))




def actor(client, payload):
    global resume_flag
    transitions, all_log_prob = [], 0.0
    (obs, _, _), done = env.reset(task_name=payload["task_name"], now_seed=payload["now_seed"], execute=True, instruction=payload["instruction"]), False
    while not done:
        if resume_flag:
            for _ in range(config.per_update_steps):
                actions = client.request("get_actions", obs)
                while actions is None:
                    time.sleep(2)
                    actions = client.request("get_actions", obs)
                while "wait" in actions:
                    time.sleep(5)
                    actions = client.request("get_actions", obs)
                
                next_obs, done, info = env.step(actions["action"])

                transition = dict(
                    observations=obs,
                    noises=actions['noise'],
                    log_probs=actions['log_prob'],
                    rewards=info['progress'],
                    dones=done,
                )
                obs = next_obs
                # all_log_prob += actions['log_prob']
                transitions.append(copy.deepcopy(transition))

            response = client.request("record", {"transitions": transitions})
            while response is None:
                time.sleep(2)
                response = client.request("record", {"transitions": transitions})
            transitions = []
            resume_flag = False
    
    # for transition in transitions:
    #     transition['log_probs'] = all_log_prob / len(transitions)

    # response = client.request("record", {"transitions": transitions, "progresses": info['progress']})
    # while response is None:
    #     time.sleep(2)
    #     response = client.request("record", {"transitions": transitions, "progresses": info['progress']})
    env.close_env()


##############################################################################

def main():
    global resume_flag
    reset_flag, reset_cfg = False, None
    Tclient = TrainerClient(
        "actor_env",
        "localhost",
        make_trainer_config(),
        wait_for_server=True,
        timeout_ms=10000,
    )

    def receive_seed(payload):
        nonlocal reset_flag, reset_cfg
        global resume_flag
        if "exit_and_reset" in payload:
            print_yellow("Received exit_and_reset signal, exiting...")
            time.sleep(2)
            os.kill(os.getpid(), signal.SIGKILL)
        elif 'now_seed' not in payload and "resume" in payload:
            resume_flag = True
        else:
            reset_flag, reset_cfg = True, payload
        return
        print_blue(f"Received seed: {payload['now_seed']}")
        actor(Tclient, payload)
        print_green(f"Finished seed: {payload['now_seed']}")
    
    Tclient.recv_network_callback(receive_seed)
    response = Tclient.request("register", {})
    while response is None:
        time.sleep(2)
        response = Tclient.request("register", {})

    print_red("starting actor loop")
    while True:
        if reset_flag:
            print_blue(f"Received seed: {reset_cfg['now_seed']}")
            actor(Tclient, reset_cfg)
            print_green(f"Finished seed: {reset_cfg['now_seed']}")
            reset_flag, reset_cfg = False, None
        


if __name__ == "__main__":
    print_yellow("Starting actor...")
    main()



