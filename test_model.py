from stable_baselines3 import PPO
from env import DinoEnv
import time

if __name__ == "__main__":
    env = DinoEnv()

    model = PPO.load("dino_ppo_model", device="cpu")

    obs = env.reset()[0]

    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        print(action)

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        time.sleep(0.05)

    print("Game Over! Total reward:", total_reward)
    env.close()
