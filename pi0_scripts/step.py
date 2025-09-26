#!/usr/bin/env python3
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
import glob
import tqdm
from absl import app, flags
from natsort import natsorted
import subprocess
import yaml
from agentlace.inference import InferenceClient
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent

import sys
sys.path.insert(0, str(root_path))

from pi0.src.openpi.training.rl_cfg import rl_config, RoboTwinEnv

config = rl_config()
FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("checkpoint_path", str(root_path / "checkpoints"), "Path to save checkpoints.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")


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

    Iclient = InferenceClient('10.91.1.35', 6379)

    output_file = open(str(root_path / "eval_results" / "eval_results3" / "policy_results.log"), "a")
    save_dir = root_path / "eval_results" / "eval_results3"
    save_dir.mkdir(parents=True, exist_ok=True)
    camera_config_path = str(root_path / "task_config" / "_camera_config.yml")
    with open(camera_config_path, "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)["D435"]
    video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
    eval_count = 0

    env = RoboTwinEnv(root_path=root_path)

    buffer_dir = os.path.join(FLAGS.checkpoint_path, "buffer")
    buffer_files = natsorted(glob.glob(os.path.join(buffer_dir, "transitions_*.pkl")))
    if FLAGS.checkpoint_path and os.path.exists(buffer_dir) and buffer_files:
        start_step = int(os.path.basename(buffer_files[-1])[12:-4]) + 1
    else:
        start_step = 0

    update_steps = 0
    # transitions = []
    # demo_transitions = []
    
    obs, instruction, now_seed = env.reset(save_video=True)
    env.task.eval_video_path = save_dir
    ffmpeg = return_ffmpeg(save_dir, video_size, eval_count)
    env.task._set_eval_video_ffmpeg(ffmpeg)

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        try:
            actions = Iclient.call("prediction", obs)
            if actions is None or actions.shape != (50, 14):
                raise ValueError(f"Step {step}: actions interface failed")

            next_obs, reward, done, info = env.step(actions)
            
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                dones=done,
            )
            obs = next_obs
            
            # transitions.append(copy.deepcopy(transition))
            '''if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))'''
            
            if done:
                result_msg = f"progress: {info['progress']}, task_name: {instruction}, seed: {now_seed}"
                output_file.write(f"{update_steps}-{eval_count}: {result_msg}\n")
                output_file.flush()
                env.task._del_eval_video_ffmpeg()
                env.close_env(env.task)
                eval_count += 1
                obs, instruction, now_seed = env.reset(save_video=True)
                env.task.eval_video_path = save_dir
                ffmpeg = return_ffmpeg(save_dir, video_size, eval_count)
                env.task._set_eval_video_ffmpeg(ffmpeg)

            response = Iclient.call("record", (transition, info))
            if response is None:
                raise ValueError(f"Step {step}: record interface failed")
            else:
                update_steps = response

        except Exception as e:
            print_red(f"Step {step}: Exception occurred: {e}")
            continue 


##############################################################################

def main(_):
    print_green("starting actor loop")
    actor()

if __name__ == "__main__":
    app.run(main)



