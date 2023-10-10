import gymnasium
from gymnasium import spaces
import pygame
import numpy as np
from gymnasium.envs.registration import register


class GridWorldEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, num_agents=2):
        self.size = size  # The size of the square grid
        self.num_agents = num_agents
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        num_values = num_agents * 4
        self.observation_space = spaces.Box(0, size - 1, shape=(num_values,), dtype=float)

        self.lidarRange = 8
        self.lidar_sweep_res = (np.arctan2(1, self.lidarRange)%np.pi ) * 2
        self.lidar_step_res = 1
    
        self._num_resets = 0

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        action_space = []
        for i in range(num_agents):
            action_space.append(4)
        self.action_space = spaces.MultiDiscrete(action_space)

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
        observation = []
        for i in range(self.num_agents):
            agent_location = self._agent_location[i*2:i*2+2]
            closest_unvisited = self.closest_non_visited(self._agent_location[i*2:i*2+2])
            observation = np.concatenate((observation, agent_location, closest_unvisited))
        return observation

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location[0:2] - self.closest_non_visited(self._agent_location[0:2]), ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self._num_resets += 1
        print(self._num_resets)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=self.num_agents * 2, dtype=int)
        self._previous_location = self.np_random.integers(0, self.size, size=self.num_agents * 2, dtype=int)
        self._two_previous_location = self.np_random.integers(0, self.size, size=self.num_agents * 2, dtype=int)

        self._visited = np.zeros((self.size, self.size))
        self._unknown = 0
        self._frontier = 1
        self._known_empty = 2
        self._known_wall = 3
        self._ground_truth = np.ones((self.size, self.size)) * self._known_empty

        # # Place a wall in the middle of the ground truth grid
        # self._ground_truth[self.size // 2, :] = self._known_wall

        for i in range(0, self.num_agents * 2, 2):
            self._visited[self._agent_location[i]][self._agent_location[i+1]] = 1
        self._total_steps = 0

        observation = self._get_obs()
        info = self._get_info()
        # print("observation", observation)

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self._two_previous_location = self._previous_location
        self._previous_location = self._agent_location.copy()

        # Move the agent, but prohibit moving through walls and moving outside the grid
        for i in range(0, self.num_agents):
            agent_action = action[i]
            proposed_location = np.clip(self._agent_location[i*2:i*2+2] + self._action_to_direction[agent_action], 0, self.size - 1)
            if self._ground_truth[proposed_location[0]][proposed_location[1]] != self._known_wall:
                self._agent_location[i*2:i*2+2] = np.clip(
                    self._agent_location[i*2:i*2+2] + self._action_to_direction[agent_action], 0, self.size - 1
                )


        # for i in range(0, self.num_agents):
        #     agent_action = action[i]
        #     proposed_location = self._agent_location[i*2:i*2+2] + self._action_to_direction[agent_action]
        #     if self._ground_truth[proposed_location[0]][proposed_location[1]] != self._known_wall:
        #         self._agent_location[i*2:i*2+2] = np.clip(
        #             self._agent_location[i*2:i*2+2] + self._action_to_direction[agent_action], 0, self.size - 1
        #         )

        # # Move the agent
        # for i in range(0, self.num_agents):
        #     agent_action = action[i]
        #     self._agent_location[i*2:i*2+2] = np.clip(
        #         self._agent_location[i*2:i*2+2] + self._action_to_direction[agent_action], 0, self.size - 1
        #     )

        # reward = 0
        # # reward agents for being in the middle
        # for i in range(0, self.num_agents * 2, 2):
        #     if(self._agent_location[i] == 3 and self._agent_location[i+1] == 3):
        #         reward += 1
        #     else:
        #         reward += 0

        # Reward is 1 if we explored the closest space
        # reward = 0
        # for i in range(0, self.num_agents * 2, 2):
        #     explored_closest = self._agent_location[i:i+2] == self.closest_non_visited(self._previous_location[i:i+2])
        #     reward += 1 if explored_closest.all() else 0

        self._total_steps += 1
        # # # Reward is 1 if we explored a new space
        # # reward = 0
        # # for i in range(0, self.num_agents * 2, 2):
        # #     previously_unvisited = self._visited[self._agent_location[i]][self._agent_location[i+1]] == 0
        # #     reward += 1 if previously_unvisited else 0

        # # double the reward if all agents explored the closest space
        # if reward == self.num_agents:
        #     reward *= 2

        # Mark the current location as visited
        # for i in range(0, self.num_agents * 2, 2):
        #     self._visited[self._agent_location[i]][self._agent_location[i+1]] = 1

        reward = self.scan()

        # Penalize moving away from the next closest location to explore
        # if(self.calculate_distance(self._agent_location, self.closest_non_visited()) > self.calculate_distance(self._previous_loation, self._previous_goal)):
        #     reward = -.5
        # Penalize staying in the same place
        for i in range(0, self.num_agents * 2, 2):
            if((self._agent_location[i:i+2] == self._previous_location[i:i+2]).all()):
                reward -= .5
            elif((self._agent_location[i:i+2] == self._two_previous_location[i:i+2]).all()):
                reward -= 1


        print(reward)
        # Finish if all elements are known wall or known empty
        terminated = np.all(
            (self._visited == self._known_wall) | (self._visited == self._known_empty)
        )
        reward += 100 if terminated else 0

        # terminated = terminated or self._total_steps >= (100 + 50 * self._num_resets)
        # if(terminated):
        #     reward += 10
            # if(self._total_steps <= ((self.size * self.size) / self.num_agents) + 2):
            #     reward += 30

        # print("reward: ", reward)
        # print("left unvisited: ", np.sum(self._visited == 0))

        # self._previous_loation = self._agent_location
        # self._previous_goal = self.closest_non_visited()
        observation = self._get_obs()
        # print("observation", observation)
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        print("observation", observation)
        return observation, reward, terminated, False, info

    def closest_non_visited(self, agent_location):
        # Find the locations where the grid is unknown
        unvisited_locations = np.argwhere(self._visited == self._unknown)
        
        # Calculate the Manhattan distance to each non-one location
        distances = np.abs(unvisited_locations - np.array(agent_location))

        # Calculate the sum of distances along axis 1 to get Manhattan distances
        manhattan_distances = np.sum(distances, axis=1)

        if(len(manhattan_distances) == 0):
            return [0,0]
        # Find the index of the location with the minimum Manhattan distance
        closest_location_index = np.argmin(manhattan_distances)

        # Get the closest location
        return unvisited_locations[closest_location_index]
    
    def closest_wall(self, agent_location):
        # Find the locations where the known map is a wall
        wall_locations = np.argwhere(self._ground_truth == self._known_wall)

        # Calculate the Manhattan distance to each non-one location
        distances = np.abs(wall_locations - np.array(agent_location))

        # Calculate the sum of distances along axis 1 to get Manhattan distances
        manhattan_distances = np.sum(distances, axis=1)

        if(len(manhattan_distances) == 0):
            return [0,0]
        # Find the index of the location with the minimum Manhattan distance
        closest_location_index = np.argmin(manhattan_distances)

        # Get the closest location
        return wall_locations[closest_location_index]
    
    def calculate_distance(self, location_one, location_two):
        return np.linalg.norm(
            location_one - location_two, ord=1
        )
    

    # def _scan_area(self):
    #     for i in range(0, self.num_agents):
    #         for x in range(self._agent_location[i*2] - self._sensor_range, self._agent_location[i*2] + self._sensor_range + 1):
    #             for y in range(self._agent_location[i*2+1] - self._sensor_range, self._agent_location[i*2+1] + self._sensor_range + 1):
    #                 if(x >= 0 and x < self.size and y >= 0 and y < self.size):
    #                     self._visited[x][y] = 1

    def scan(self):
        total_learned = 0
        for agent_num in range(0, self.num_agents):
            for i, angle in enumerate(np.arange(0, 2*np.pi, self.lidar_sweep_res)):
                position_x = self._agent_location[agent_num*2]
                position_y = self._agent_location[agent_num*2+1]

                ray_cast_samples = np.arange(0,self.lidarRange, self.lidar_step_res)
                for j, r in enumerate(ray_cast_samples):
                    # get the point rounded to the nearest grid
                    x = int(np.round(position_x + r*np.sin(angle)))
                    y = int(np.round(position_y + r*np.cos(angle)))

                    if x < 0 or x >= self.size or y < 0 or y >= self.size:
                        break
                    sampled_point= self._ground_truth[x][y]
                    if sampled_point == self._known_wall:# obstacle
                        self._visited[x][y] = self._known_wall
                        break
                    if r == max(ray_cast_samples):# frontier
                        if self._ground_truth[x][y] == self._known_empty:
                            break
                        self._visited[x][y] = self._frontier
                        break
                    # free space
                    if self._visited[x][y] != self._known_empty:
                        total_learned += 1
                    self._visited[x][y] = self._known_empty
        return total_learned
                    

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

        # First we draw the visited locations
        for x in range(self.size):
            for y in range(self.size):
                if self._visited[x][y] == self._known_empty:
                    pygame.draw.rect(
                        canvas,
                        (0, 255, 0),
                        pygame.Rect(
                            pix_square_size * np.array([x, y]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

        # Next we draw the frontier locations
        for x in range(self.size):
            for y in range(self.size):
                if self._visited[x][y] == self._frontier:
                    pygame.draw.rect(
                        canvas,
                        (255, 255, 0),
                        pygame.Rect(
                            pix_square_size * np.array([x, y]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

        # Next we draw the wall locations
        for x in range(self.size):
            for y in range(self.size):
                if self._visited[x][y] == self._known_wall:
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
                        pygame.Rect(
                            pix_square_size * np.array([x, y]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

        # Now we draw the agent
        for i in range(self.num_agents):
            pygame.draw.circle(
                canvas,
                (i*25, 0, 255),
                (self._agent_location[i*2:i*2+2] + 0.5) * pix_square_size,
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