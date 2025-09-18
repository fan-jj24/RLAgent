#!/usr/bin/env python3
import os 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.97"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from flax.training import checkpoints
from pathlib import Path
root_path = Path(__file__).resolve().parent
'''Root path: /home/pop.fan/Pi0-RL-RoboTwin'''
import sys
sys.path.insert(0, str(root_path))

from agentlace.trainer import TrainerServer

from serl_launcher.utils.launcher import make_trainer_config, make_wandb_logger
from serl_launcher.data.data_store import SimpleReplayBufferDataStore
from serl_launcher.agents.RLAgent import Agent
from rl_cfg import rl_config

config = rl_config()
FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("checkpoint_path", str(root_path / "checkpoints"), "Path to save checkpoints.")
flags.DEFINE_boolean("debug", False, "Debug mode.")  

seeds_file = "seed_174458.json"
with open(seeds_file, "r") as f:
    seeds = json.load(f)

def print_green(x):
    print("\033[92m {}\033[00m".format(x))
def print_red(x):
    print("\033[91m {}\033[00m".format(x))
def print_yellow(x):
    print("\033[93m {}\033[00m".format(x))
def print_blue(x):
    print("\033[94m {}\033[00m".format(x))

#############################################################################


def learner(rng, agent, replay_buffer, wandb_logger):
    latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)) if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path) else None
    if latest_ckpt is not None:
        start_step = int(os.path.basename(latest_ckpt)[11:]) + 1
        print(f"Resuming from checkpoint at step {start_step}.")
    else:
        start_step = 0
        print("No checkpoint found. Starting from step 0.")

    env_num, traj, advs, update_step, env_step, seed_flag, train_flag, reset_step = 0, [], [], 0, 0, True, False, 0

    def stats_callback(type: str, payload: dict) -> dict:
        nonlocal rng, env_num, traj, advs, train_flag, agent

        if type == "register":
            env_num += 1
            return {"success": True}
        
        elif type == "get_actions":
            if train_flag: return {"wait": True}
            rng, seed = jax.random.split(rng)
            action, noise, log_prob = agent.get_actions(payload, seed)
            action = np.asarray(jax.device_get(action))
            noise = np.asarray(jax.device_get(noise))
            log_prob = np.asarray(jax.device_get(log_prob))
            return {"action": action, "noise": noise, "log_prob": log_prob}
        
        elif type == "record":
            traj.append(payload["transitions"])
            # advs.append(payload["progresses"])
            return {"success": True}
    
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.start(threaded=True)
    time.sleep(30)
    print_red(f"{env_num} actors are connected.")

    def wait_to_train():
        nonlocal advs, traj, env_step, seed_flag, train_flag
        replay_buffer.clear()

        while len(traj) != env_num :
            time.sleep(0.01)
            
        train_flag = True
        returns = np.zeros(shape=(env_num, config.per_update_steps), dtype=np.float32)
        for i in range(env_num):
            for j in range(config.per_update_steps-1, -1, -1):
                returns[i,j] = traj[i][j]['rewards'] + config.discount * (0 if j==config.per_update_steps-1 else returns[i,j+1])
        advs = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-8)
        for i in range(env_num):
            for j in range(config.per_update_steps):
                traj[i][j]["advantages"] = advs[i,j]
                replay_buffer.insert(traj[i][j])
        if traj[0][-1]["dones"]:
            actor_log = []
            for i in range(env_num):
                actor_log.append(traj[i][-1]["rewards"])
            actor_log = jnp.array(actor_log)
            wandb_logger.log({"Actor": {f"mean_{env_step//200}": jnp.mean(actor_log), f"std_{env_step//200}": jnp.std(actor_log)}}, step=env_step%200)
            seed_flag = True
            env_step += 1
        # adv_values = np.asarray(advs)
        # adv_mean, adv_std = adv_values.mean(), adv_values.std()
        # wandb_logger.log({f"mean_{env_step//200}": adv_mean, f"std_{env_step//200}": adv_std}, step=env_step%200)
        # adv_values = (adv_values - adv_mean) / (adv_std + 1e-8)
        # for index, values in enumerate(traj):
        #     for value in values:
        #         value["advantages"] = adv_values[index]
        #         replay_buffer.insert(value)
        advs, traj = [], []
########################   learning loop   ###################################

    for step in range(start_step, 300000):
        if seed_flag: 
            server.publish_network(seeds[(env_step%200)//2])
            print(f"Published network for seed: {seeds[(env_step%200)//2]['now_seed']}")
            seed_flag = False
        server.publish_network({"resume": True})
        wait_to_train()
        for _ in range(config.iters):
            start_idx = -1
            while start_idx < len(replay_buffer) - 1:
                batch, start_idx = replay_buffer.sample_sequential(config.batch_size, start_idx + 1)
                agent, info = agent.update(batch)
                wandb_logger.log({"train": info}, step=update_step)
                update_step += 1
                reset_step += 1
        train_flag = False

        if seed_flag and reset_step // config.ACTOR_RESTART_INTERVAL > 0:
            env_num, reset_step = 0, 0
            print_blue(f"Step {step}: Sending restart command to all actors.")
            server.publish_network({"exit_and_reset": True})
            time.sleep(10)
            print_red("Waiting for actors to reconnect...")
            time.sleep(300)
            print_red(f"{env_num} actors reconnected.")

        # if step > 0 and step % config.checkpoint_period == 0:
        #     checkpoints.save_checkpoint(os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=1)

##############################################################################

def main(_):

    agent = Agent.create_pixels(
        rng=jax.random.PRNGKey(FLAGS.seed),
        pretrained_policy_path=str(root_path / "params_0"),
        low_bound = config.low_bound,
        high_bound = config.high_bound,
        entropy_coef = config.entropy_coef,
        kl_div_coef = config.kl_div_coef,
        target_kl_div = config.target_kl_div,
        temperature_coef = config.temperature_coef,
    ) 

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = SimpleReplayBufferDataStore(
            config.observation_space,
            config.action_space,
            capacity=config.replay_buffer_capacity,
        )
        wandb_logger = make_wandb_logger(
            project="pi0-rlhf-gspo",
            description="pi0gspo",
            tag=["pi0", "rlhf", "post-training"],
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()


    print_green("starting learner loop")
    learner(
        jax.random.PRNGKey(os.getpid()),
        agent,
        replay_buffer,
        wandb_logger,
    )




if __name__ == "__main__":
    app.run(main)
