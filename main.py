import gym_examples
from gym_examples.wrappers import RelativePosition
import gym
import pygame
from gym.utils.play import play

# mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1, (pygame.K_UP,): 2, (pygame.K_DOWN,): 3}
# play(gym.make('gym_examples/GridWorld-v0', render_mode = 'human'), keys_to_action=mapping)

env = gym.make('gym_examples/GridWorld-v0', render_mode = 'human', size=5)
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()