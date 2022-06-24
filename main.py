from gym.utils import seeding
from gym import spaces
import numpy as np
import pygame
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from prey import Prey
from predator import Predator
from grass import Grass
import time

if __name__ == '__main__':

    n_iterations = 100000
    n_learning_cycles = 100
    cycle = 0
    epsilon = 0.1
    seed_nr = 1  # None
    x_size = 10
    n_initial_predators = 6
    n_initial_prey = 6
    n_initial_grass = 6

    initial_energy_level_predators = 5
    initial_energy_level_prey = 4
    initial_energy_level_grass = 2

    predators_metabolism_energy = 0
    prey_metabolism_energy = 0.2
    predators_step_energy = 0
    prey_step_energy = 0.1  # 1
    predators_reproduction_energy_minimum = 5
    prey_reproduction_energy_minimum = 4
    grass_reproduction_energy_minimum = 2.5
    grass_growth = 0.1
    prey_max_energy_consumption = 1.5  # maximum number of grass energy eaten if available

    predators_reproduction_rate = 0.1
    predators_reproduction_age = 10
    predators_death_rate = 0.019
    predators_cost_of_reproduction = 4

    prey_reproduction_rate = 0.1
    prey_reproduction_age = 8
    prey_death_rate = 0.05
    prey_cost_of_reproduction = 1

    grass_reproduction_rate = 0.02  # energy increase per time unit (= full cycle)

    # position pygame window
    import os

    x = 1280
    y = 0
    sleep_time = 0
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x, y)
    pygame.init()  # Initializing Pygame


    class PredatorPreyGridEnv(AECEnv):  # noqa
        # "noqa" https://stackoverflow.com/questions/34066933/pycharm-warning-must-implement-all-abstract-methods

        n_agent_types = 3
        metadata = {'render.modes': ['human']}

        def __init__(self, _x_size, _n_initial_predators, _n_initial_prey, _n_initial_grass):

            super().__init__()
            # seeding random generators
            self.np_random = None
            self.seed(seed_nr)
            # initial grid set up
            self.x_size = _x_size
            self.n_initial_predators = _n_initial_predators
            self.n_initial_prey = _n_initial_prey
            self.n_initial_grass = _n_initial_grass

            self.observations = None
            self._cumulative_rewards = dict()
            self.rewards = dict()
            self.infos = dict()
            self.dones = dict()
            self.agents_name_to_instance_dict = dict()  # links "agent" (agents name) to the actual agent
            self.agent_selection = None
            self._agent_selector = None

            # location
            self.n_agents_in_grid_cells = np.zeros((self.n_agent_types, self.x_size, self.x_size), dtype=int)
            self.n_agents_in_grid = np.zeros(self.n_agent_types, dtype=int)  # [0 for _ in range(self.n_agent_types)]
            self.agent_names_in_grid_cells = [[[[] for _ in range(self.x_size)] for _ in range(self.x_size)]
                                              for _ in range(self.n_agent_types)]

            self._action_to_direction = {0: np.array([-1, -1]), 3: np.array([0, -1]), 6: np.array([1, -1]),
                                         1: np.array([-1, 0]), 4: np.array([0, 0]), 7: np.array([1, 0]),
                                         2: np.array([-1, 1]), 5: np.array([0, 1]), 8: np.array([1, 1])}
            """
            Maps actions into relative grid positions of the Moore neighborhood
            The following dictionary maps abstract actions from `self.action_space` to 
            the direction we will walk in if that action is taken.
            I.e. 0 corresponds to "right", 1 to "up" etc.
            """
            self.n_neighborhood_cells = len(self._action_to_direction)

            # observation space
            self.obs_range = _x_size  # obs_range: Size of the box around the agent that the agent observes
            self.obs_shape = [self.obs_range, self.obs_range, self.n_agent_types]
            self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float32)
            self.obs_range_predators = _x_size  # full environment
            self.obs_range_prey = _x_size  # full environment

            # agent definitions
            self.predator_type_nr = 0
            self.prey_type_nr = 1
            self.grass_type_nr = 2
            self.agents = []  # predator_0, ...,predator_n-1, prey_n,..., prey_n+m, grass_n+m+1, ...]
            self.id_nr = 0
            self.n_all_moving_agents = self.n_initial_predators + self.n_initial_prey
            self.n_all_possible_agents = 10000  # maximum number of agents
            self.n_actions_agents = self.n_neighborhood_cells  # Moore
            self.act_space = spaces.Discrete(self.n_actions_agents)
            self.action_spaces = [self.act_space for _ in range(self.n_all_moving_agents)]  # magent: action_spaces_list
            self.n_features = 12

            self.predators_metabolism_energy = predators_metabolism_energy
            self.predators_step_energy = predators_step_energy
            self.predators_reproduction_energy_minimum = predators_reproduction_energy_minimum
            self.predators_cost_of_reproduction = predators_cost_of_reproduction

            self.prey_max_energy_consumption = prey_max_energy_consumption
            self.prey_metabolism_energy = prey_metabolism_energy
            self.prey_step_energy = prey_step_energy
            self.prey_reproduction_energy_minimum = prey_reproduction_energy_minimum
            self.prey_cost_of_reproduction = prey_cost_of_reproduction

            self.grass_growth = grass_growth
            self.grass_reproduction_energy_minimum = grass_reproduction_energy_minimum

            # reinforcement learning
            self.epsilon = epsilon
            self.n_iterations = n_iterations
            self.cycle = cycle
            self.n_learning_cycles = n_learning_cycles
            self.learning_rate = 0.05
            self.discount_factor = 1

            # rendering
            self.pixel_scale = 100
            self.screen = None

        def instance(self, _agent_name):
            return self.agents_name_to_instance_dict[_agent_name]

        def get_neighborhood_location(self, _x, _y):
            n_agents_in_neighborhood = np.zeros([self.n_agent_types, self.n_neighborhood_cells], dtype=int)
            moore_index = 0
            for d_x in [-1, 0, 1]:
                for d_y in [-1, 0, 1]:
                    x_neighbor_location = (_x + d_x) % self.x_size
                    y_neighbor_location = (_y + d_y) % self.x_size
                    n_agents_in_neighborhood[self.predator_type_nr][moore_index] = \
                        self.n_agents_in_grid_cells[self.predator_type_nr][x_neighbor_location, y_neighbor_location]
                    n_agents_in_neighborhood[self.prey_type_nr][moore_index] = \
                        self.n_agents_in_grid_cells[self.prey_type_nr][x_neighbor_location, y_neighbor_location]
                    n_agents_in_neighborhood[self.grass_type_nr][moore_index] = \
                        self.n_agents_in_grid_cells[self.grass_type_nr][x_neighbor_location, y_neighbor_location]
                    moore_index += 1
            return n_agents_in_neighborhood

        def get_features(self, _agent_name, _x, _y):
            agent_instance = self.instance(_agent_name)
            features = np.zeros(self.n_features)
            n_agents_in_neighborhood = self.get_neighborhood_location(_x, _y)  # 3 x 9 matrix
            n_predators_in_grid = self.n_agents_in_grid[self.predator_type_nr]
            n_prey_in_grid = self.n_agents_in_grid[self.prey_type_nr]
            n_grass_in_grid = self.n_agents_in_grid[self.grass_type_nr]
            n_predators_in_neighborhood = sum(n_agents_in_neighborhood[self.predator_type_nr][:])
            n_prey_in_neighborhood = sum(n_agents_in_neighborhood[self.prey_type_nr][:])
            n_grass_in_neighborhood = sum(n_agents_in_neighborhood[self.grass_type_nr][:])
            match agent_instance.agent_type_nr:
                case self.predator_type_nr:
                    if n_prey_in_grid != 0:
                        features[0] = n_prey_in_neighborhood / n_prey_in_grid
                    else:
                        features[0] = 0
                    if n_predators_in_grid != 0:
                        features[1] = n_predators_in_neighborhood / n_predators_in_grid
                    else:
                        features[1] = 0
                    if n_grass_in_grid != 0:
                        features[2] = n_grass_in_neighborhood / n_grass_in_grid
                    else:
                        features[2] = 0
                    if n_prey_in_neighborhood != 0:
                        for i in range(self.n_neighborhood_cells):
                            features[i + 3] = n_agents_in_neighborhood[self.prey_type_nr][i] / n_prey_in_neighborhood
                    else:
                        for i in range(self.n_neighborhood_cells):
                            features[i + 3] = 0
                case self.prey_type_nr:
                    if n_predators_in_grid != 0:
                        features[0] = n_predators_in_neighborhood / n_predators_in_grid
                    else:
                        features[0] = 0
                    if n_prey_in_grid != 0:
                        features[1] = n_prey_in_neighborhood / n_prey_in_grid
                    else:
                        features[1] = 0
                    if n_grass_in_grid != 0:
                        features[2] = n_grass_in_neighborhood / n_grass_in_grid
                    else:
                        features[2] = 0
                    if n_predators_in_neighborhood != 0:
                        for i in range(self.n_neighborhood_cells):
                            features[i + 3] = n_agents_in_neighborhood[self.predator_type_nr][i] / \
                                              n_predators_in_neighborhood
                    else:
                        for i in range(self.n_neighborhood_cells):
                            features[i + 3] = 0
            return features

        def is_cell_occupied_with_grass(self, _x, _y):
            is_occupied = self.n_agents_in_grid_cells[self.grass_type_nr][_x, _y] > 0
            return is_occupied

        def step_energy(self, _agent_name, _action):
            step_energy = 0
            if _action == 4:  # no move
                return step_energy
            agent_instance = self.instance(_agent_name)
            match agent_instance.agent_type_nr:
                case self.predator_type_nr:
                    step_energy = self.predators_step_energy
                case self.predator_type_nr:
                    step_energy = self.prey_step_energy
            return step_energy

        def move_agent(self, _agent_name, new_x, new_y):
            agent_instance = self.instance(_agent_name)
            match agent_instance.agent_type_nr:
                case self.predator_type_nr:
                    predator = agent_instance
                    old_x = predator.x
                    old_y = predator.y
                    predator.x = new_x
                    predator.y = new_y
                    self.n_agents_in_grid_cells[self.predator_type_nr][old_x, old_y] -= 1
                    self.n_agents_in_grid_cells[self.predator_type_nr][new_x, new_y] += 1
                    self.agent_names_in_grid_cells[self.predator_type_nr][old_x][old_y].remove(_agent_name)
                    self.agent_names_in_grid_cells[self.predator_type_nr][new_x][new_y].append(_agent_name)
                case self.prey_type_nr:
                    prey = agent_instance
                    old_x = prey.x
                    old_y = prey.y
                    prey.x = new_x
                    prey.y = new_y
                    self.n_agents_in_grid_cells[self.prey_type_nr][old_x, old_y] -= 1
                    self.n_agents_in_grid_cells[self.prey_type_nr][new_x, new_y] += 1
                    self.agent_names_in_grid_cells[self.prey_type_nr][old_x][old_y].remove(_agent_name)
                    self.agent_names_in_grid_cells[self.prey_type_nr][new_x][new_y].append(_agent_name)
                case self.grass_type_nr:
                    return

        def execute_action_selected_agent(self, _action):
            _agent_name = self.agent_selection
            agent_instance = self.instance(_agent_name)
            new_x = (agent_instance.x + self._action_to_direction[_action][0]) % self.x_size
            new_y = (agent_instance.y + self._action_to_direction[_action][1]) % self.x_size
            self.move_agent(agent_instance.agent_name, new_x, new_y)
            step_energy = self.step_energy(_agent_name, action)
            return step_energy

        def remove_agent(self, _agent_name):
            # removes agent
            agent_instance = self.instance(_agent_name)
            assert self.dones[
                _agent_name
            ], "an agent that was not done as attempted to be removed"
            del self.dones[_agent_name]
            del self.rewards[_agent_name]
            del self._cumulative_rewards[_agent_name]
            del self.infos[_agent_name]
            self.agents.remove(_agent_name)
            self.agent_names_in_grid_cells[agent_instance.agent_type_nr][agent_instance.x][agent_instance.y].remove(
                _agent_name)

            self.n_agents_in_grid_cells[agent_instance.agent_type_nr][agent_instance.x, agent_instance.y] -= 1
            self.agents_name_to_instance_dict.pop(_agent_name, None)
            self.n_agents_in_grid[agent_instance.agent_type_nr] -= 1
            #print(_agent_name + " is removed")
            return self.agents

        # noinspection PyTypeChecker
        def create_agent(self, agent_type_nr, new_x, new_y):
            _agent_name = None
            match agent_type_nr:
                case self.predator_type_nr:
                    predator = Predator(new_x, new_y, self.id_nr)
                    type_name = predator.agent_type_name
                    _agent_name = f"{type_name}_{self.id_nr}"
                    predator.agent_name = _agent_name
                    predator.energy_level = initial_energy_level_predators
                    predator.age = 0
                    # localization helpers
                    self.n_agents_in_grid_cells[self.predator_type_nr][new_x, new_y] += 1
                    self.agent_names_in_grid_cells[self.predator_type_nr][new_x][
                        new_y].append(_agent_name)
                    self.agents.append(_agent_name)
                    self.agents_name_to_instance_dict[_agent_name] = predator
                    self.n_agents_in_grid[self.predator_type_nr] += 1
                    self.dones[_agent_name] = False
                    self.rewards[_agent_name] = 0
                    self._cumulative_rewards[_agent_name] = 0
                    self.infos[_agent_name] = {}
                    self.id_nr += 1
                case self.prey_type_nr:
                    prey = Prey(new_x, new_y, self.id_nr)
                    type_name = prey.agent_type_name
                    _agent_name = f"{type_name}_{self.id_nr}"
                    prey.agent_name = _agent_name
                    prey.energy_level = initial_energy_level_prey
                    prey.age = 0
                    # localization helpers
                    self.n_agents_in_grid_cells[self.prey_type_nr][new_x, new_y] += 1
                    self.agent_names_in_grid_cells[self.prey_type_nr][new_x][
                        new_y].append(_agent_name)
                    self.agents.append(_agent_name)
                    self.agents_name_to_instance_dict[_agent_name] = prey
                    self.n_agents_in_grid[self.prey_type_nr] += 1
                    self.dones[_agent_name] = False
                    self.rewards[_agent_name] = 0
                    self._cumulative_rewards[_agent_name] = 0
                    self.infos[_agent_name] = {}
                    self.id_nr += 1
                case self.grass_type_nr:
                    grass = Grass(new_x, new_y, self.id_nr)
                    type_name = grass.agent_type_name
                    _agent_name = f"{type_name}_{self.id_nr}"
                    grass.agent_name = _agent_name
                    grass.energy_level = initial_energy_level_grass
                    grass.age = 0
                    # localization helpers
                    self.n_agents_in_grid_cells[self.grass_type_nr][new_x, new_y] += 1
                    if self.n_agents_in_grid_cells[self.grass_type_nr][new_x, new_y] > 1:
                        print("ERROR: NOT MORE THAN ONE GRASS AGENT PER CELL")
                    self.agent_names_in_grid_cells[self.grass_type_nr][new_x][
                        new_y].append(_agent_name)
                    self.agents.append(_agent_name)
                    self.agents_name_to_instance_dict[_agent_name] = grass
                    self.n_agents_in_grid[self.grass_type_nr] += 1
                    self.dones[_agent_name] = False
                    self.rewards[_agent_name] = 0
                    self._cumulative_rewards[_agent_name] = 0
                    self.infos[_agent_name] = {}
                    self.id_nr += 1
            return _agent_name

        def seed(self, seed=None):
            self.np_random, np_seed = seeding.np_random(seed)

        def get_random_location(self):
            _x = self.np_random.randint(0, self.x_size)
            _y = self.np_random.randint(0, self.x_size)
            return _x, _y

        def get_neighborhood_agents(self, _agent_name):
            _x = self.instance(_agent_name).x
            _y = self.instance(_agent_name).y
            n_agents_in_neighborhood = np.zeros([self.n_agent_types, self.n_neighborhood_cells], dtype=int)
            moore_index = 0
            for d_x in [-1, 0, 1]:
                for d_y in [-1, 0, 1]:
                    x_neighbor_location = (_x + d_x) % self.x_size
                    y_neighbor_location = (_y + d_y) % self.x_size
                    n_agents_in_neighborhood[self.predator_type_nr][moore_index] = \
                        self.n_agents_in_grid_cells[self.predator_type_nr][x_neighbor_location, y_neighbor_location]
                    n_agents_in_neighborhood[self.prey_type_nr][moore_index] = \
                        self.n_agents_in_grid_cells[self.prey_type_nr][x_neighbor_location, y_neighbor_location]
                    n_agents_in_neighborhood[self.grass_type_nr][moore_index] = \
                        self.n_agents_in_grid_cells[self.grass_type_nr][x_neighbor_location, y_neighbor_location]
                    moore_index += 1
            return n_agents_in_neighborhood

        def get_empty_grass_location(self):
            grass_instance = self.instance(self.agent_selection)
            x_grass = grass_instance.x
            y_grass = grass_instance.y
            possible_grass_offspring_moore_index_list = []
            moore_index = 0
            for d_x in [-1, 0, 1]:
                for d_y in [-1, 0, 1]:
                    x_neighbor_location = (x_grass + d_x) % self.x_size
                    y_neighbor_location = (y_grass + d_y) % self.x_size
                    if not self.is_cell_occupied_with_grass(x_neighbor_location, y_neighbor_location):
                        possible_grass_offspring_moore_index_list.append(moore_index)
                    moore_index += 1

            if possible_grass_offspring_moore_index_list:
                random_moore_index = self.np_random.randint(0, len(possible_grass_offspring_moore_index_list))
                random_moore_location = possible_grass_offspring_moore_index_list[random_moore_index]
                _x = (grass_instance.x + self._action_to_direction[random_moore_location][0]) % self.x_size
                _y = (grass_instance.y + self._action_to_direction[random_moore_location][1]) % self.x_size
                return _x, _y
            else:
                return None

        def get_reward(self, _agent_name):
            """
            reward = opponent * type + 2 * same * type
            opponent = number of the other species type within the agent’s Moore
            neighborhood normalized by the number of that species in the world
            type is 1 for predator and −1 for prey
            same = {0, 1} for if the opponent is on the same location
            """
            _reward = 0
            agent_instance = self.instance(_agent_name)
            n_agents_in_neighborhood = self.get_neighborhood_agents(_agent_name)
            n_predators_in_grid = self.n_agents_in_grid[self.predator_type_nr]
            n_prey_in_grid = self.n_agents_in_grid[self.prey_type_nr]
            n_predators_in_neighborhood = sum(n_agents_in_neighborhood[self.predator_type_nr][:])
            n_prey_in_neighborhood = sum(n_agents_in_neighborhood[self.prey_type_nr][:])
            center_position = int((self.n_neighborhood_cells - 1) / 2)
            if type(agent_instance) is Predator:
                if n_prey_in_grid != 0:
                    relative_prey_in_neighborhood = n_prey_in_neighborhood / n_prey_in_grid
                else:
                    relative_prey_in_neighborhood = 0
                if n_agents_in_neighborhood[self.prey_type_nr][center_position] > 0:  # also prey in center position
                    _reward = relative_prey_in_neighborhood + 2
                else:
                    _reward = relative_prey_in_neighborhood
            elif type(agent_instance) is Prey:
                if n_predators_in_grid != 0:
                    relative_predators_in_neighborhood = n_predators_in_neighborhood / n_predators_in_grid
                else:
                    relative_predators_in_neighborhood = 0
                if n_agents_in_neighborhood[self.predator_type_nr][center_position] > 0:  # also predator in center
                    _reward = -relative_predators_in_neighborhood - 2
                else:
                    _reward = -relative_predators_in_neighborhood
            elif type(agent_instance) is Grass:
                _reward = 0
            return _reward

        def observe(self, _agent_name):
            _observation = self.n_agents_in_grid_cells
            return _observation

        def reset(self, seed=None):
            if seed is not None:
                self.seed(seed=seed)
                # seeding self.act_space.sample()
                self.act_space.seed(seed=seed)

            self.observations = {_agent_name: None for _agent_name in self.agents}
            for j in range(self.n_initial_predators):
                _x, _y = self.get_random_location()
                predator_name = self.create_agent(self.predator_type_nr, _x, _y)
                predator_instance = self.instance(predator_name)
                predator_instance.weights = self.np_random.rand(self.n_features) * 6 - 3  # initial weights
            for j in range(self.n_initial_prey):
                _x, _y = self.get_random_location()
                prey_name = self.create_agent(self.prey_type_nr, _x, _y)
                # print(prey_name+" created")
                prey_instance = self.instance(prey_name)
                prey_instance.weights = self.np_random.rand(self.n_features) * 6 - 3  # initial weights
                # print("prey weights "+str(prey_instance.weights))
            for j in range(self.n_initial_grass):
                _x, _y = self.get_random_location()
                while self.is_cell_occupied_with_grass(_x, _y):
                    _x, _y = self.get_random_location()
                self.create_agent(self.grass_type_nr, _x, _y)

            self._agent_selector = agent_selector(self.agents)
            self.agent_selection = self._agent_selector.next()

        def last(self, observe=True):
            _agent_name = self.agent_selection
            _agent_instance = self.instance(_agent_name)
            _agent_instance.is_learning = True  # self.cycle > self.n_learning_cycles
            _done = self.dones[agent_name]
            _info = self.infos[agent_name]
            _reward = self.get_reward(agent_name)
            _observation = self.observe(agent_name)
            if _agent_instance.is_learning and type(_agent_instance) is not Grass:
                _agent_instance.update_weight(self, _reward, _agent_instance.q)
                if _agent_name == "predator_1":
                    #print("weights agent " + _agent_name)
                    print(_agent_instance.weights)
                    print("Q = "+str(_agent_instance.q))

            return _observation, _reward, _done, _info

        def state(self):
            current_state = self.n_agents_in_grid_cells
            return current_state

        def render(self, mode="human"):
            self.screen = pygame.display.set_mode(
                (self.pixel_scale * self.x_size, self.pixel_scale * self.x_size))
            # draw grid
            grid_color = (255, 255, 255)
            grid_border_color = (0, 0, 0)
            predator_color = (255, 0, 0, 10)  # red
            prey_color = (0, 0, 255)  # blue
            grass_color = (0, 128, 0, 50)  # green
            self.screen.fill(grid_color)
            for i in range(self.x_size):
                for j in range(self.x_size):
                    pos = pygame.Rect(self.pixel_scale * i, self.pixel_scale * j,
                                      self.pixel_scale, self.pixel_scale)
                    pygame.draw.rect(self.screen, grid_border_color, pos, 3)
            # draw agent counts
            font = pygame.font.SysFont('Comic Sans MS', self.pixel_scale * 2 // 5)
            for i in range(self.x_size):
                for j in range(self.x_size):
                    (pos_x, pos_y) = (self.pixel_scale * i + self.pixel_scale // 1.3,
                                      self.pixel_scale * j + self.pixel_scale // 1.7)
                    prey_count = self.n_agents_in_grid_cells[self.prey_type_nr][i, j]
                    count_text: str
                    if prey_count < 1:
                        count_text = ""
                    elif prey_count < 10:
                        count_text = str(prey_count)
                    else:
                        count_text = "+"
                    text = font.render(count_text, False, prey_color)
                    self.screen.blit(text, (pos_x, pos_y))
                    predator_count = self.n_agents_in_grid_cells[self.predator_type_nr][i, j]
                    count_text: str
                    if predator_count < 1:
                        count_text = ""
                    elif predator_count < 10:
                        count_text = str(predator_count)
                    else:
                        count_text = "+"
                    text = font.render(count_text, False, predator_color)
                    self.screen.blit(text, (pos_x, pos_y - self.pixel_scale // 2))
            # draw agents
            for _agent_name in self.agents:
                scale = 1
                if _agent_name == self.agent_selection:
                    scale = 2
                agent_instance = self.instance(_agent_name)
                _x = agent_instance.x
                _y = agent_instance.y
                if agent_instance.agent_type_name == "predator":

                    center = (int(self.pixel_scale * _x + self.pixel_scale / 2),
                              int(self.pixel_scale * _y + self.pixel_scale / 2))
                    pygame.draw.circle(self.screen, predator_color, center,
                                       int(self.pixel_scale * scale / 4))
                    # id_nr
                    text = font.render(str(agent_instance.id_nr), False, (255, 255, 225))
                    size_id_nr = len(str(agent_instance.id_nr))
                    center_id = (int(self.pixel_scale * _x + self.pixel_scale / (2 + (size_id_nr * 5) / 10)),
                                 int(self.pixel_scale * _y + self.pixel_scale / 2.5))
                    self.screen.blit(text, center_id)

                elif agent_instance.agent_type_name == "prey":

                    center = (int(self.pixel_scale * _x + self.pixel_scale / 2),
                              int(self.pixel_scale * _y + self.pixel_scale / 2))
                    pygame.draw.circle(self.screen, prey_color, center,
                                       int(self.pixel_scale * scale / 4))
                    # id_nr
                    text = font.render(str(agent_instance.id_nr), False, (255, 255, 225))
                    size_id_nr = len(str(agent_instance.id_nr))
                    center_id = (int(self.pixel_scale * _x + self.pixel_scale / (2 + (size_id_nr * 5) / 10)),
                                 int(self.pixel_scale * _y + self.pixel_scale / 2.5))
                    self.screen.blit(text, center_id)
                elif agent_instance.agent_type_name == "grass":
                    # color='lightgreen'
                    pos = pygame.Rect(self.pixel_scale * _x + 6, self.pixel_scale * _y + 6,
                                      self.pixel_scale - 12, self.pixel_scale - 12)
                    pygame.draw.rect(self.screen, grass_color, pos, 10 * scale)
                    # id_nr
                    text = font.render(str(agent_instance.id_nr), False, grass_color)
                    size_id_nr = len(str(agent_instance.id_nr))
                    center_id = (int(self.pixel_scale * _x + self.pixel_scale / (6 + (size_id_nr * 5) / 10)),
                                 int(self.pixel_scale * _y + self.pixel_scale / 10))
                    self.screen.blit(text, center_id)

            pygame.display.update()

        def get_random_agent_grid_cell(self, agent_type_nr, _x, _y):
            agent_instance = None
            match agent_type_nr:
                case self.prey_type_nr:
                    prey_name_list = self.agent_names_in_grid_cells[self.prey_type_nr][_x][_y]
                    if prey_name_list:  # prey available at new predator location?
                        random_prey_name = self.np_random.choice(prey_name_list)
                        agent_instance = self.instance(random_prey_name)
                case self.grass_type_nr:
                    grass_name = self.agent_names_in_grid_cells[self.grass_type_nr][_x][_y]

                    if grass_name:  # grass available at new prey location?
                        agent_instance = self.instance(grass_name[0])
            return agent_instance

        def step(self, _action):
            if self.dones[self.agent_selection]:
                self._was_done_step(_action)
                return
            # main contents of step
            agent_instance = self.instance(self.agent_selection)
            match agent_instance.agent_type_nr:
                case self.predator_type_nr:
                    predator = agent_instance
                    predator.age += 1
                    predator.energy_level -= self.predators_metabolism_energy
                    predator.energy_level -= self.execute_action_selected_agent(_action)
                    if predator.energy_level <= 0:
                        self.dones[predator.agent_name] = True
                        #print(predator.agent_name + " has no energy left and is done")
                    elif predator.energy_level <= self.predators_reproduction_energy_minimum:
                        eaten_prey_instance = self.get_random_agent_grid_cell(self.prey_type_nr, predator.x, predator.y)
                        # eaten_prey_instance = self.instance(eaten_prey_instance.agent_name)
                        if eaten_prey_instance:
                            self.dones[eaten_prey_instance.agent_name] = True
                            #print("eaten_prey_instance.agent_name " + str(eaten_prey_instance.agent_name))
                            predator.energy_level += eaten_prey_instance.energy_level
                            #print(predator.agent_name + " is eating " + eaten_prey_instance.agent_name +
                            #      " and has energy-level " + str(predator.energy_level))
                    elif predator.energy_level > self.predators_reproduction_energy_minimum:
                        # energy level above "predators_reproduction_energy_minimum"
                        predator.energy_level -= self.predators_cost_of_reproduction
                        new_predator_name = self.create_agent(self.predator_type_nr, predator.x, predator.y)
                        #print(new_predator_name + " created from " + predator.agent_name)
                        new_predator_instance = self.instance(new_predator_name)
                        new_predator_instance.weights = predator.weights
                case self.prey_type_nr:
                    prey = agent_instance
                    prey.age += 1
                    prey.energy_level -= self.prey_metabolism_energy
                    prey.energy_level -= self.execute_action_selected_agent(_action)
                    if prey.energy_level <= 0:
                        self.dones[prey.agent_name] = True
                        #print(prey.agent_name + " has no energy left and is done")
                    elif prey.energy_level <= self.prey_reproduction_energy_minimum:
                        # grass available for prey with positive energy level?
                        eaten_grass_instance = self.get_random_agent_grid_cell(self.grass_type_nr, prey.x, prey.y)
                        if eaten_grass_instance:
                            prey_real_energy_consumption = min(self.prey_max_energy_consumption,
                                                               eaten_grass_instance.energy_level)
                            prey.energy_level += prey_real_energy_consumption
                            eaten_grass_instance.energy_level -= prey_real_energy_consumption
                            if eaten_grass_instance.energy_level <= 0:
                                self.dones[eaten_grass_instance.agent_name] = True
                                #print(prey.agent_name + " is eating " + eaten_grass_instance.agent_name +
                                #      " and has energy-level " + str(prey.energy_level) + ", grass dies ")
                    elif prey.energy_level > self.prey_reproduction_energy_minimum:
                        prey.energy_level -= self.prey_cost_of_reproduction
                        new_prey_name = self.create_agent(self.prey_type_nr, prey.x, prey.y)
                        #print(new_prey_name + " created from " + prey.agent_name)
                        new_prey_instance = self.instance(new_prey_name)
                        new_prey_instance.weights = prey.weights
                case self.grass_type_nr:
                    grass = agent_instance
                    grass.energy_level += self.grass_growth
                    if grass.energy_level > self.grass_reproduction_energy_minimum \
                            and not self.get_empty_grass_location() is None:
                        _x, _y = self.get_empty_grass_location()
                        new_grass_name = self.create_agent(self.grass_type_nr, _x, _y)
                        #print(new_grass_name + " created @ [" + str(_x) + "," + str(
                        #    _y) + "] from " + grass.agent_name + " @ [" + str(grass.x) + "," + str(grass.y) + "]")

            self.agent_selection = self._agent_selector.next()
            self.agent_selection = self._dones_step_first()

        # noinspection PyProtectedMember
        def _was_done_step(self, _action):
            """
            Helper function that performs step() for done agents.
            The sequence of living agents can give rise to unexpected behavior
            in the following cases of "done" and subsequent removal:

                1) the selected agent causes another agent to "done" where the other agent has
                already been selected earlier in the agent_order of the same cycle,
                2) the selected agent causes itself to "done"
                3) the selected agents causes its direct successor in the agent_order to "done".

            """
            if _action is not None:
                raise ValueError("when an agent is done, the only valid action is None")
            # to fix a runtime error when self._skip_agent_selection is already being "done"
            # and deleted by the selected agent: see 3)
            if self.agent_selection == self._skip_agent_selection:
                self._skip_agent_selection = self._agent_selector.next()

            # removes done agent
            _agent_name = self.agent_selection
            assert self.dones[
                _agent_name
            ], "an agent that was not done as attempted to be removed"
            #print("FRONT LOADING")
            self.remove_agent(_agent_name)

            # finds next done agent or loads next live agent (Stored in _skip_agent_selection)
            _dones_order = [_agent_name for _agent_name in self.agents if self.dones[_agent_name]]
            if _dones_order:
                if getattr(self, "_skip_agent_selection", None) is None:
                    self._skip_agent_selection = self.agent_selection
                self.agent_selection = _dones_order[0]
            else:
                if getattr(self, "_skip_agent_selection", None) is not None:
                    self.agent_selection = self._skip_agent_selection
                self._skip_agent_selection = None
            self._clear_rewards()
            # to fix changing array indexes when the selected agent
            # deletes an earlier agent in the agent order (agents): see 1)
            # and to fix when the selected agent deletes itself: see 2)
            if self.agents and self.agents[self._agent_selector._current_agent - 1] != \
                    self.agent_selection:
                self._agent_selector._current_agent -= 1

        @property
        def agent_selector(self):
            """
            To fix problem: "Access to a protected member _agent_selector of a class"
            """
            return self._agent_selector


    env = PredatorPreyGridEnv(x_size,
                              n_initial_predators,
                              n_initial_prey,
                              n_initial_grass)


    def policy(_observation, _agent_name):

        def neighborhood_evaluation(_env, _x, _y):
            """
            Evaluate the neighboring cells
            """
            moore_index = 0
            _score = np.zeros(_env.n_neighborhood_cells)
            for d_x in [-1, 0, 1]:
                for d_y in [-1, 0, 1]:
                    x_eval = (_x + d_x) % _env.x_size
                    y_eval = (_y + d_y) % _env.x_size
                    f_i = _env.get_features(_agent_name, x_eval, y_eval)
                    _agent_instance = _env.instance(_env.agent_selection)
                    cell_score = np.dot(f_i, _agent_instance.weights)
                    _score[moore_index] = cell_score
                    moore_index += 1
            return _score

        agent_instance = env.instance(_agent_name)
        # observation_space = env.observation_spaces(agent)
        _action = None
        if agent_instance.agent_type_nr == 2:  # grass
            _action = 4  # no move
        else:

            r = env.np_random.rand()

            if r < 1 - env.epsilon:  # exploitation
                score = neighborhood_evaluation(env, agent_instance.x, agent_instance.y)
                best_score_index = np.argmax(score[:])  # select the line with the best score
                _action = best_score_index
                agent_instance.q = score[best_score_index]
            else:  # exploration
                _action = env.act_space.sample()
        return _action


    env.reset(seed_nr)
    iteration = 0
    env.cycle = 0
    for agent_name in env.agent_iter(env.n_iterations):
        if env.agent_selector.is_first():
            env.cycle += 1
            print("cycle " + str(env.cycle))
            if env.cycle > 100:
                break
        if env.agent_selection == "predator":
            print("agent_selection " + env.agent_selection + " energy: " +
                  str(env.instance(env.agent_selection).energy_level))
            print(env.agents)
            print()
        observation, reward, done, info = env.last()
        action = policy(observation, agent_name) if not done else None
        env.step(action)
        #env.render()
        time.sleep(sleep_time)
        iteration += 1

    input('Press any key to exit\n')
