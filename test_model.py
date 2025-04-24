import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from env import DinoEnv
import time

def evaluate_model(model_path, num_episodes=20):
    # 1) Load the trained policy
    model = DQN.load(model_path, device="cpu")

    scores = []
    for ep in range(num_episodes):
        env = DinoEnv()
        obs, _ = env.reset()
        done = False
        total_reward = 0

        # 2) Run one full episode
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

        scores.append(total_reward)
        env.close()
        print(f"Episode {ep+1}: score = {total_reward:.2f}")

    return scores

if __name__ == "__main__":
    model_file = "dino_dqn_model_basic_500k"  # adjust if needed
    num_episodes = 20

    scores = evaluate_model(model_file, num_episodes)

    # 3) Plot the scores
    episodes = np.arange(1, num_episodes+1)
    plt.plot(episodes, scores)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (Distance)")
    plt.title(f"Model Performance over {num_episodes} Episodes")
    plt.grid(True)
    plt.show()