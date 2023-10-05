import gymnasium
from gymnasium import spaces
import pygame
import numpy as np
from gymnasium.envs.registration import register


class GridWorldEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(0, size - 1, shape=(8,), dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.MultiDiscrete([4,4])

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        self._total_steps = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return np.concatenate([self._agent_location[0:2], self.closest_non_visited(self._agent_location[0:2]), self._agent_location[2:4], self.closest_non_visited(self._agent_location[2:4])])

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location[0:2] - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=4, dtype=int)
        # self._previous_loation = self._agent_location
        # self._previous_goal = self._agent_location

        # # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location[0:2]
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )

        self._visited = np.zeros((self.size, self.size))
        self._visited[self._agent_location[0]][self._agent_location[1]] = 1
        self._total_steps = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        assert len(action) == 2, "Two actions are required"
        
        agent1_action = action[0]
        agent2_action = action[1]

        # print("action ", agent1_action)
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction_one = self._action_to_direction[agent1_action]
        direction_two = self._action_to_direction[agent2_action]
        # print("direction ", direction_one)
        # print("direction 2 ", direction_two)

        previous_agent_2_location = self._agent_location[2:4]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location[0:2] + direction_one, 0, self.size - 1
        )
        self._agent_location = np.append(self._agent_location, 
                                         np.clip(previous_agent_2_location + direction_two, 0, self.size - 1))
        # An episode is done iff the agent has reached the target
        self._total_steps += 1
        
        # Mark the current location as visited
        self._visited[self._agent_location[0]][self._agent_location[1]] = 1
        self._visited[self._agent_location[2]][self._agent_location[3]] = 1

        # Reward is 1 if we explored a new space
        previously_unvisited = self._visited[self._agent_location[0]][self._agent_location[1]] == 0
        reward = 1 if previously_unvisited else 0

        second_previously_unvisited = self._visited[self._agent_location[2]][self._agent_location[3]] == 0
        reward += 1 if second_previously_unvisited else 0

        # Penalize moving away from the next closest location to explore
        # if(self.calculate_distance(self._agent_location, self.closest_non_visited()) > self.calculate_distance(self._previous_loation, self._previous_goal)):
        #     reward = -.5
        # # Penalize staying in the same place
        # if((self._agent_location == self._previous_loation).all()):
        #     reward  = -1
        terminated = np.all(self._visited == 1)
        print("reward: ", reward)
        print("left unvisited: ", np.sum(self._visited == 0))

        # self._previous_loation = self._agent_location
        # self._previous_goal = self.closest_non_visited()
        observation = self._get_obs()
        print(observation)
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def closest_non_visited(self, agent_location):
        # Find the locations where the grid is not equal to 1
        unvisited_locations = np.argwhere(self._visited != 1)

        # Calculate the Manhattan distance to each non-one location
        distances = np.abs(unvisited_locations - np.array(agent_location))

        # Calculate the sum of distances along axis 1 to get Manhattan distances
        manhattan_distances = np.sum(distances, axis=1)

        if(len(manhattan_distances) == 0):
            return tuple([0,0])
        # Find the index of the location with the minimum Manhattan distance
        closest_location_index = np.argmin(manhattan_distances)

        # Get the closest location
        return tuple(unvisited_locations[closest_location_index])
    
    def calculate_distance(self, location_one, location_two):
        return np.linalg.norm(
            location_one - location_two, ord=1
        )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        for x in range(self.size):
            for y in range(self.size):
                if self._visited[x][y] == 1:
                    pygame.draw.rect(
                        canvas,
                        (0, 255, 0),
                        pygame.Rect(
                            pix_square_size * np.array([x, y]),
                            (pix_square_size, pix_square_size),
                        ),
                    )
        # pygame.draw.rect(
        #     canvas,
        #     (255, 0, 0),
        #     pygame.Rect(
        #         pix_square_size * self._target_location,
        #         (pix_square_size, pix_square_size),
        #     ),
        # )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location[0:2] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Now we draw the second agent
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self._agent_location[2:4] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# Example for the CartPole environment
register(
    # unique identifier for the env `name-version`
    id="GridWorld-v0",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=GridWorldEnv,
)