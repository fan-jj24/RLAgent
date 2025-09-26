#!/usr/bin/env python3
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
import tqdm
import datetime
import json
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent

import sys
sys.path.insert(0, str(root_path))

from pi0.src.openpi.training.rl_cfg import RoboTwinEnv





def actor():

    logs_dir = root_path / "seed_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%H%M%S")
    log_path = logs_dir / f"seed_{ts}.json"
    env = RoboTwinEnv(root_path=root_path)

    pbar = tqdm.tqdm(range(0, 100000), dynamic_ncols=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        try:
            for step in pbar:
                (obs, instruction, now_seed) = env.reset()
                task_name = env.args['task_name']
                rec = {
                    "task_name": task_name,
                    "instruction": instruction,
                    "now_seed": int(now_seed),
                }
                if not first:
                    f.write(",\n")
                json.dump(rec, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
                first = False
                env.close_env(env.task)
        except KeyboardInterrupt:
            print("\n捕获到 Ctrl+C, 正在安全关闭 JSON 文件...")
        finally:
            f.write("\n]\n")
            f.flush()
            os.fsync(f.fileno())
        

actor()




