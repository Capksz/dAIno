from env import DinoEnv
import time
import random
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os, glob

def make_env(name, rank):
    filename = os.path.join("eval_train", f"{name}_monitor_{rank}.csv")
    return lambda: Monitor(DinoEnv(), filename=filename)

def find_latest_checkpoint(path, prefix):
    # find files like prefix_<step>_steps.zip
    files = glob.glob(os.path.join(path, f"{prefix}_*.zip"))
    if not files:
        return None
    # sort by timestep in filename
    files.sort(key=lambda fn: int(fn.split('_')[-2]))
    return files[-1]

if __name__ == "__main__":
    name = "literature2-1_100k"

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("tensorboard", exist_ok=True)
    os.makedirs("eval_train", exist_ok=True)
    num_envs = 6
    env_fns = [make_env(name, i) for i in range(num_envs)]
    env = DummyVecEnv(env_fns)

    ckpt = find_latest_checkpoint("checkpoints", "dino_dqn_checkpoint")
    if ckpt:
        print("Loading checkpoint:", ckpt)
        model = DQN.load(
            ckpt,
            env=env,
            device="cpu",
            tensorboard_log="tensorboard/"
        )
        reset_timesteps = False
    else:
        print("Starting fresh training")
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
            tensorboard_log="tensorboard/"
        )
        reset_timesteps = True

    # Create a callback that saves the model every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="checkpoints/",
        name_prefix="dino_dqn_checkpoint"
    )

    # Evaluation callback (deterministic eval every 20k steps)
    eval_env = DummyVecEnv([make_env(name, "eval")])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints/",
        log_path="tensorboard/",
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    # Train
    model.learn(
        total_timesteps=100_000,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=reset_timesteps
    )

    # Save
    model.save(f"dino_dqn_model_{name}")
    env.close()
    eval_env.close()
