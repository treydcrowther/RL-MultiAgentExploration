from stable_baselines3 import A2C  # Import the A2C algorithm
from stable_baselines3 import PPO  # Import the PPO algorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import GridWorldEnv
import gymnasium


vec_env = DummyVecEnv([lambda: gymnasium.make("GridWorld-v0", size=50, num_agents=8, render_mode="human")])
# Parallel environments
model = PPO.load("GridWorldModel")

num_episodes = 10
for episode in range(num_episodes):
    obs  = vec_env.reset()
    done = False
    total_reward = 0

    while not done:
        # Use the loaded model to select an action
        action, _ = model.predict(obs, deterministic=True)

        # Take the action in the environment
        obs, reward, done, _ = vec_env.step(action)

        # Accumulate the reward
        total_reward += reward

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")