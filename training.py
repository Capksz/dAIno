from env import DinoEnv
import time
import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv


def make_env():
    def _init():
        return DinoEnv()
    return _init

if __name__ == "__main__":
    num_envs = 32  # This makes num_env number of browsers for faster training (can be reduced)
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    model = PPO("MlpPolicy", env, device="cpu", verbose=1)
    model.learn(total_timesteps=500_000)
    model.save("dino_ppo_model")
    env.close()

    # Training Continuation if needed
    # model = PPO.load("dino_ppo_model", env=env, device="cpu")
    #
    # model.learn(total_timesteps=500_000)
    #
    # model.save("dino_ppo_model_continued")
