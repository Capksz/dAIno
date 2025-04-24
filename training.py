from env import DinoEnv
import time
import random
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import glob
import os

def make_env():
    def _init():
        return DinoEnv()
    return _init

def find_latest_checkpoint(path, prefix):
    # find files like prefix_<step>_steps.zip
    files = glob.glob(os.path.join(path, f"{prefix}_*.zip"))
    if not files:
        return None
    # sort by timestep in filename
    files.sort(key=lambda fn: int(fn.split('_')[-2]))
    return files[-1]

if __name__ == "__main__":
    num_envs = 6
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    ckpt_path = find_latest_checkpoint("./checkpoints", "dino_dqn_checkpoint")
    if ckpt_path:
        print("Loading checkpoint:", ckpt_path)
        model = DQN.load(ckpt_path, env=env, device="cpu")
        # continue training without resetting the timestep counter
        reset_timesteps = False
    else:
        print("No checkpoint found; starting fresh.")

        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-4,
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            device="cpu",
            verbose=1,
        )
        reset_timesteps = True

    # Create a callback that saves the model every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,  # Save every 10,000 steps
        save_path='./checkpoints/',  # Folder to save checkpoints
        name_prefix='dino_dqn_checkpoint'  # Prefix for checkpoint files
    )

    model.learn(total_timesteps=500_000, callback=checkpoint_callback,reset_num_timesteps=reset_timesteps)
    model.save("dino_dqn_model_distancebase_500k")

    # next to train
    # model.learn(total_timesteps=100_000, callback=checkpoint_callback)
    # model.save("dino_dqn_model_literature2_100k")
    env.close()
