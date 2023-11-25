import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import GridWorldEnv
from os.path import exists 

# if __name__ == "__main__":
#     # vec_env = SubprocVecEnv([lambda: gym.make("GridWorld-v0", size=50, num_agents=1) for _ in range(8)])
#     vec_env = DummyVecEnv([lambda: gym.make("GridWorld-v0", size=10, num_agents=1)])

#     # check_env(env, warn=True, skip_render_check=True)

#     model = PPO("MlpPolicy", vec_env, verbose=1)
#     model.learn(total_timesteps=500_000)

#     model.save("TestFullInformation2Agents")

if __name__ == "__main__":

    if(not exists("TestFullInformation2Agents.zip")):
        vec_env = SubprocVecEnv([lambda: gym.make("GridWorld-v0", size=15, num_agents=2) for _ in range(8)])

        print("recreating")
        # model = PPO("MlpPolicy", vec_env, verbose=1)
        model = PPO("MlpPolicy", vec_env, verbose=1)

        model.learn(total_timesteps=100_000)

        model.save("TestFullInformation2Agents")
    else:
        print("not recreating")

    for i in range(500):
        vec_env = SubprocVecEnv([lambda: gym.make("GridWorld-v0", size=15, num_agents=2) for _ in range(8)])

        # model = PPO("MlpPolicy", vec_env, verbose=1)
        model = PPO.load("TestFullInformation2Agents", vec_env)

        model.learn(total_timesteps=100_000)

        model.save("TestFullInformation2Agents")