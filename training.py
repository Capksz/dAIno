from env import DinoEnv
import time
import random
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


def make_env():
    def _init():
        return DinoEnv()
    return _init

if __name__ == "__main__":
    num_envs = 1 # This makes num_env number of browsers for faster training (can be reduced)
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    model = PPO("MlpPolicy", env, device="cpu", verbose=1)
    # model = DQN("MlpPolicy", env, device="cpu", verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path='./checkpoints/',
        name_prefix='dino_model'
    )

    model.learn(total_timesteps=100_000, callback=checkpoint_callback)
    model.save("dino_ppo_model_final")
    env.close()

    # Training Continuation if needed
    # model = DQN.load("dino_ppo_model_continued_1,000,000", env=env, device="cpu")
    #
    # model.learn(total_timesteps=500_000)
    #
    # model.save("dino_ppo_model_continued_1,500,000")
