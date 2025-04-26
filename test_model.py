import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from env import DinoEnv
import time
import os

def evaluate_model(model_path, num_episodes=100):
    # 1) Load the trained policy
    model = DQN.load(model_path, device="cpu")

    result = []
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

        # After the episode is done, fetch the on-screen score:
        game_score = env.driver.execute_script("""
            // Grab the array of displayed digits, join them into a string, then parse to int
            return parseInt(Runner.instance_.distanceMeter.digits.join(''), 10);
        """)
        result.append((total_reward, game_score))
        env.close()
        print(f"Episode {ep+1}: total_reward = {total_reward:.2f}, game_score = {game_score:.2f}")

    return result

if __name__ == "__main__":
    model_file = "dino_dqn_model_literature2.1_100k" # change to your model
    num_episodes = 100

    base = os.path.basename(model_file) 
    name = base.replace("dino_", "").replace("_model", "")
    
    results = evaluate_model(model_file, num_episodes)
    total_rewards = [res[0] for res in results]
    game_scores   = [res[1] for res in results]
    episodes      = np.arange(1, num_episodes + 1)

    # create the figure and first axis
    fig, ax1 = plt.subplots()
    color1 = 'tab:blue'
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward", color=color1)
    ax1.plot(episodes, total_rewards, marker='o', color=color1, label="Total Reward")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(min(total_rewards) - 10, max(total_rewards) + 10)

    # create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel("Game Score", color=color2)
    ax2.plot(episodes, game_scores, marker='s', color=color2, label="Game Score")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(min(game_scores)   - 5,  max(game_scores)   + 5)

    # add grid, title, and legend
    fig.tight_layout()
    plt.title(f"{name} Model Performance over {num_episodes} Episodes")
    ax1.grid(True)

    # combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # save it to files
    output_dir = "eval_test"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join("eval_test", f"{name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved evaluation plot to {out_path}")