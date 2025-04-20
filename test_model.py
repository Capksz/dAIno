from stable_baselines3 import PPO
from env import DinoEnv
import time

if __name__ == "__main__":
    # Create a single environment (no SubprocVecEnv needed)
    env = DinoEnv()

    # Load the saved model
    model = PPO.load("dino_ppo_model", device="cpu")

    obs = env.reset()[0]  # Gymnasium returns (obs, info)

    total_reward = 0
    done = False

    while not done:
        # Model predicts action
        action, _ = model.predict(obs, deterministic=True)

        # Step in environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Slow down to watch it in real time (optional)
        time.sleep(0.05)

    print("Game Over! Total reward:", total_reward)
    env.close()
