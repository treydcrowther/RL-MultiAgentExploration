import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import GridWorldEnv


if __name__ == "__main__":
    # vec_env = SubprocVecEnv([lambda: gym.make("GridWorld-v0", size=50, num_agents=1) for _ in range(8)])
    vec_env = DummyVecEnv([lambda: gym.make("GridWorld-v0", size=50, num_agents=8)])

    # check_env(env, warn=True, skip_render_check=True)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=50_000)

    model.save("GridWorldModel")