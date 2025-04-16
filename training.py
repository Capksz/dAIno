from env import DinoEnv
import time
import random
import numpy as np
from stable_baselines3 import PPO

env = DinoEnv()

model = PPO("MlpPolicy", env, device="cpu", verbose=1)
model.learn(total_timesteps=100_000)
