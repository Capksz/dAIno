import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO
from env import DinoEnv
import time

def run_model(model_path, type):
    if(type == "DQN"):
      model = DQN.load(model_path, device="cpu")
    else:
      model = PPO.load(model_path, device="cpu")

    env = DinoEnv()
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

    env.close()


