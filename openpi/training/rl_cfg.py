# import importlib
# import yaml
# import os
import gymnasium as gym
import numpy as np
# import random
# import jax
# import pdb
# from pi0.src.openpi.transforms import Normalize, Unnormalize, make_bool_mask, pad_to_dim
# from pi0.src.openpi.shared import image_tools, normalize
# from pi0.src.openpi.models import tokenizer

# from envs.utils.create_actor import UnStableError
# from description.utils.generate_episode_instructions import generate_episode_descriptions

# import logging
# logging.getLogger("curobo").setLevel(logging.ERROR)

action_std = np.array([0.18549174070358276,
        0.5489271283149719,
        0.45225954055786133,
        0.31335774064064026,
        0.05495546758174896,
        0.27847668528556824,
        0.3967568278312683,
        0.1887737661600113,
        0.5549138784408569,
        0.46206119656562805,
        0.31976795196533203,
        0.06079776957631111,
        0.2986145317554474,
        0.4085076153278351,])

class rl_config:
    def __init__(self, action_chunk = 50):
        self.action_space = gym.spaces.Box(
            -np.pi, np.pi, shape=(action_chunk,14), dtype=np.float32  
        )
        self.image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                            -np.pi, np.pi, shape=(14,), dtype =np.float32
                        ),  #joints vector
                    
                "image": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8) 
                                for key in self.image_keys}
                ),
                "image_mask": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 1, shape=(1,), dtype=np.bool_) for key in self.image_keys}
                ),
                "tokenized_prompt": gym.spaces.Box(
                    0, 257151, shape = (48,), dtype=np.int32
                ),
                "tokenized_prompt_mask": gym.spaces.Box(
                    0, 1, shape = (48,), dtype=np.bool_
                ),
            }
        )

        self.batch_size = 32
        self.discount = 0.97
        self.soft_target_update_rate = 0.005
        self.target_policy_noise = action_std * 0.01
        self.noise_clip = self.target_policy_noise * 2.0
        self.augmentation_function = None
        self.reward_bias = 0.0
        self.lora = False

        self.replay_buffer_capacity = 10000

        self.max_steps = 1000000
        self.buffer_period = 100
        self.log_period = 10
        self.training_starts = 100
        self.cta_ratio = 10
        self.steps_per_update = 50
        self.checkpoint_period = 100
    
'''def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


class RoboTwinEnv():
    def __init__(self, root_path = None):
        self.task_name = ["stack_blocks_two", "stack_blocks_three", "stack_bowls_two", "stack_bowls_three",]
        task_config = "demo_randomized"
        config_path = str(root_path / "task_config" / f"{task_config}.yml") 
        with open(config_path, "r", encoding="utf-8") as f:
            self.args = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.args['task_config'] = task_config
        
        embodiment_type = self.args.get("embodiment")
        embodiment_config_path = str(root_path / "task_config" / "_embodiment_config.yml") 
        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        def get_embodiment_file(embodiment_type):
            robot_file = _embodiment_types[embodiment_type]["file_path"]
            robot_file = str(root_path / robot_file)
            if robot_file is None:
                raise "missing embodiment files"
            return robot_file
        
        def get_embodiment_config(robot_file):
            robot_config_file = os.path.join(robot_file, "config.yml")
            with open(robot_config_file, "r", encoding="utf-8") as f:
                embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
            return embodiment_args
        

        if len(embodiment_type) == 1:
            self.args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            self.args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
            self.args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            self.args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            self.args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
            self.args["embodiment_dis"] = embodiment_type[2]
            self.args["dual_arm_embodied"] = False
        else:
            raise "number of embodiment config parameters should be 1 or 3"

        self.args["left_embodiment_config"] = get_embodiment_config(self.args["left_robot_file"])
        self.args["right_embodiment_config"] = get_embodiment_config(self.args["right_robot_file"])
        self.args["save_video"] = False

        norm_stats = normalize.load(str(root_path))
        if norm_stats is not None:
            self.input_transform = Normalize(norm_stats, use_quantiles=False)
            self.output_transform = Unnormalize(norm_stats, use_quantiles=False)
        else:
            raise ValueError("norm_stats must be provided for transforms")
        
        self.delta_action_mask = make_bool_mask(6, -1, 6, -1)
        self.tokenizer = tokenizer.PaligemmaTokenizer(48)
    
        
    def close_env(self, task):
        task.close_env(clear_cache = False)
        if self.args["render_freq"]:
            task.viewer.close()

    def reset(self, task_name = None, mode = None, save_video = False, now_seed = None, max_seed = 100000, strict = False):
        if task_name is None:
            task_name = np.random.choice(self.task_name)
        if now_seed is None:
            now_seed = random.randint(2000, 10000)

        task = class_decorator(task_name)
        self.args['task_name'] = task_name
        self.args["save_demo"] = False
        self.args["save_freq"] = 15
        while True:
            try:
                task.setup_demo(now_ep_num=0, seed=now_seed, is_test=False, **self.args)
                episode_info = task.play_once()
                s1, s2 = task.plan_success, task.check_success()
                self.close_env(task)
            except UnStableError as e:
                print("UnStableError: ", e)
                self.close_env(task)
                now_seed += 1
                continue
            except Exception as e:
                print("Exception: ",e)
                self.close_env(task)
                now_seed += 1
                pdb.set_trace()
                continue

            if (mode != "demo" or strict) and (not s1 or not s2):
                now_seed += 1
                continue
            else:
                if mode == "demo":
                    self.args["save_demo"] = True
                    self.args["save_freq"] = 8
                    if s1 and s2:
                        self.mode_flag = "success"
                    else:
                        self.mode_flag = "fail"
                else:
                    self.args["eval_mode"] = True
                    self.args["save_video"] = save_video
                self.task = task
                self.task.setup_demo(now_ep_num=0, seed=now_seed, is_test=False, **self.args)
                episode_info_list = [episode_info["info"]]
                results = generate_episode_descriptions(self.args["task_name"], episode_info_list, max_seed)
                instruction = np.random.choice(results[0]["unseen"])
                self.task.set_instruction(instruction)
                self.tokens, self.token_masks = self.tokenizer.tokenize(instruction)
                observation = self.input(self.get_observation())

                return observation, instruction, now_seed
        
    def step(self, actions):
        actions = pad_to_dim(actions, 32)
        observation = self.input(self.get_observation())
        state = observation["state"]
        state = pad_to_dim(state, 32)
        output = self.output_transform({
            "state": state,
            "actions": actions
        })
        state, actions = np.asarray(output["state"][:14]),  np.asarray(output["actions"][:, :14])
        actions[..., :14] += np.expand_dims(np.where(self.delta_action_mask, state[..., :14], 0), axis=-2)
        for action in actions:
            self.task.take_action(action)
            if self.args["save_video"]:
                self.task.get_obs() # to update the video window
        next_observation = self.input(self.get_observation())
        done = True if self.task.eval_success or self.task.take_action_cnt >= self.task.step_lim else False
        reward = 1.0 if self.task.eval_success else 0.0
        return next_observation, reward, done, {"success": reward}

    def get_observation(self):

        obs = self.task.get_obs()
        observation = {
            "state": obs["joint_action"]["vector"],
            "image": {
                "base_0_rgb": obs["observation"]["head_camera"]["rgb"],
                "left_wrist_0_rgb": obs["observation"]["left_camera"]["rgb"],
                "right_wrist_0_rgb": obs["observation"]["right_camera"]["rgb"],
            },
            "image_mask": {
                "base_0_rgb":True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": True,
            },
            "tokenized_prompt": self.tokens,
            "tokenized_prompt_mask": self.token_masks,
        }
        return observation


    def input(self, observation):
        observation["state"] = pad_to_dim(observation["state"], 32)
        if "actions" in observation:
            observation["actions"] = pad_to_dim(observation["actions"], 32)
        
        observation = self.input_transform(observation)
        
        observation["state"] = np.asarray(observation["state"][:14])  # only keep the first 14 dims
        if "actions" in observation:
            observation["actions"] = np.asarray(observation["actions"][:, :14])
        observation["image"] = {k: image_tools.resize_with_pad(v, 224, 224) for k, v in observation["image"].items()}
        observation["image"] = {k: np.array(jax.device_put(v, device=jax.devices("cpu")[0])) for k, v in observation["image"].items()}

        return observation'''

    

    
    

    

    

    




