from gym import spaces
import numpy as np
import pygame
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from prey import Prey
from predator import Predator
from grass import Grass
import random
import time

if __name__ == '__main__':

    x_size = 5
    n_iterations = 100
    seed = 10

    n_initial_predators = 3
    n_initial_prey = 3
    n_initial_grass = 3

    initial_energy_level_predators = 5
    initial_energy_level_prey = 4
    initial_energy_level_grass = 2

    predators_metabolism_energy = 0.5
    prey_metabolism_energy = 0.2  # random.uniform(0, 2)
    predators_step_energy = 0.1
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

    n_max_cycles = 10
    n_learning_iterations = 100
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

        def __init__(self, x_size, n_initial_predators, n_initial_prey, n_initial_grass):

            super().__init__()
            random.seed(seed)
            self.x_size = x_size
            self.n_initial_predators = n_initial_predators
            self.n_initial_prey = n_initial_prey
            self.n_initial_grass = n_initial_grass

            self.observations = None
            self._cumulative_rewards = dict()
            self.rewards = dict()
            self.infos = dict()
            self.dones = dict()
            self.agents_dict = dict()  # links "agent" (agents name) to the actual agent
            self.agent_selection = None
            self._agent_selector = None

            # location
            self.n_agents_in_grid_cells = np.zeros((self.n_agent_types, x_size, x_size), dtype=int)
            self.agents_lists_in_grid_cells = [
                [[[] for _ in range(PredatorPreyGridEnv.n_agent_types)] for _ in range(x_size)]
                for _ in range(x_size)]
            self.n_active_agents = [0 for _ in range(self.n_agent_types)]
            # TODO: turn dict into matrix?
            self.actions_positions_dict = {0: [-1, -1], 3: [0, -1], 6: [1, -1],
                                           1: [-1, 0], 4: [0, 0], 7: [1, 0],
                                           2: [-1, 1], 5: [0, 1], 8: [1, 1]}
            self.n_neighborhood_cells = 9
            self.relative_x_positions_neighbors = [-1, 0, 1]
            self.relative_y_positions_neighbors = [-1, 0, 1]

            # observation space
            self.obs_range = x_size  # obs_range: Size of the box around the agent that the agent observes
            self.obs_shape = [self.obs_range, self.obs_range, self.n_agent_types]
            self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float32)
            self.obs_range_predators = x_size  # full environment
            self.obs_range_prey = x_size  # full environment

            self.n_all_agents = self.n_initial_predators + self.n_initial_prey
            self.n_all_possible_agents = 10000  # maximum number of agents
            self.n_actions_agents = len(self.actions_positions_dict)  # Moore
            self.act_space = spaces.Discrete(self.n_actions_agents)
            self.action_spaces = [self.act_space for _ in range(self.n_all_agents)]  # magent: action_spaces_list
            self.n_features = 12

            # agent definitions
            self.predator_type_nr = 0
            self.prey_type_nr = 1
            self.grass_type_nr = 2
            self.agents_instance_list = []
            self.agents = []  # predator_0, ...,predator_n-1, prey_n,..., prey_n+m]
            self.id_nr = 0

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

            self.pixel_scale = 100
            self.n_iterations = n_iterations
            self.screen = None

        def instance(self, agent_name):
            return self.agents_dict[agent_name]

        def is_cell_occupied_with_grass(self, x, y):
            is_occupied = self.n_agents_in_grid_cells[self.grass_type_nr][x, y] > 0
            return is_occupied

        def step_energy(self, agent_name, action):
            step_energy = 0
            if action == 4:  # no move
                return step_energy
            agent_instance = self.instance(agent_name)
            match agent_instance.agent_type_nr:
                case self.predator_type_nr:
                    step_energy = self.predators_step_energy
                case self.predator_type_nr:
                    step_energy = self.prey_step_energy
            return step_energy

        def move_agent(self, agent_name, new_x, new_y):
            agent_instance = self.instance(agent_name)
            match agent_instance.agent_type_nr:
                case self.predator_type_nr:
                    predator = agent_instance
                    old_x = predator.x
                    old_y = predator.y
                    predator.x = new_x
                    predator.y = new_y
                    self.n_agents_in_grid_cells[self.predator_type_nr][old_x, old_y] -= 1
                    self.n_agents_in_grid_cells[self.predator_type_nr][new_x, new_y] += 1
                    self.agents_lists_in_grid_cells[old_x][old_y][self.predator_type_nr].remove(predator)
                    self.agents_lists_in_grid_cells[new_x][new_y][self.predator_type_nr].append(predator)
                case self.prey_type_nr:
                    prey = agent_instance
                    old_x = prey.x
                    old_y = prey.y
                    prey.x = new_x
                    prey.y = new_y
                    self.n_agents_in_grid_cells[self.prey_type_nr][old_x, old_y] -= 1
                    self.n_agents_in_grid_cells[self.prey_type_nr][new_x, new_y] += 1
                    self.agents_lists_in_grid_cells[old_x][old_y][self.prey_type_nr].remove(prey)
                    self.agents_lists_in_grid_cells[new_x][new_y][self.prey_type_nr].append(prey)
                case self.grass_type_nr:
                    return

        def execute_action_selected_agent(self, action):
            agent_name = self.agent_selection
            agent_instance = self.instance(agent_name)
            self.move_agent(agent_instance.agent_name,
                            (agent_instance.x + self.actions_positions_dict[action][0]) % self.x_size,
                            (agent_instance.y + self.actions_positions_dict[action][1]) % self.x_size)
            step_energy = self.step_energy(agent_name, action)
            return step_energy

        def remove_agent(self, agent_name):
            # removes agent
            agent_instance = self.instance(agent_name)
            assert self.dones[
                agent_name
            ], "an agent that was not done as attempted to be removed"
            del self.dones[agent_name]
            del self.rewards[agent_name]
            del self._cumulative_rewards[agent_name]
            del self.infos[agent_name]
            self.agents.remove(agent_name)
            self.agents_lists_in_grid_cells[agent_instance.x][
                agent_instance.y][agent_instance.agent_type_nr].remove(agent_instance)
            self.n_agents_in_grid_cells[agent_instance.agent_type_nr][
                agent_instance.x, agent_instance.y] -= 1
            self.agents_instance_list.remove(self.agents_dict[agent_name])
            self.agents_dict.pop(agent_name, None)
            self.n_active_agents[agent_instance.agent_type_nr] -= 1
            print(agent_name+" is removed")
            return self.agents

        def create_agent(self, agent_type_nr, x, y):
            agent_name = None
            match agent_type_nr:
                case self.predator_type_nr:
                    predator = Predator(x, y, self.id_nr)
                    type_name = predator.agent_type_name
                    agent_name = f"{type_name}_{self.id_nr}"
                    predator.agent_name = agent_name
                    predator.energy_level = initial_energy_level_predators
                    predator.age = 0
                    # localization helpers
                    self.n_agents_in_grid_cells[self.predator_type_nr][x, y] += 1
                    self.agents_lists_in_grid_cells[x][y][self.predator_type_nr].append(predator)
                    self.agents_instance_list.append(predator)
                    self.agents.append(agent_name)
                    self.agents_dict[agent_name] = predator
                    self.n_active_agents[self.predator_type_nr] += 1
                    self.dones[agent_name] = False
                    self.rewards[agent_name] = 0
                    self._cumulative_rewards[agent_name] = 0
                    self.infos[agent_name] = {}
                    self.id_nr += 1
                case self.prey_type_nr:
                    prey = Prey(x, y, self.id_nr)
                    type_name = prey.agent_type_name
                    agent_name = f"{type_name}_{self.id_nr}"
                    prey.agent_name = agent_name
                    prey.energy_level = initial_energy_level_prey
                    prey.age = 0
                    # localization helpers
                    self.n_agents_in_grid_cells[self.prey_type_nr][x, y] += 1
                    self.agents_lists_in_grid_cells[x][y][self.prey_type_nr].append(prey)
                    self.agents_instance_list.append(prey)
                    self.agents.append(agent_name)
                    self.agents_dict[agent_name] = prey
                    self.n_active_agents[self.prey_type_nr] += 1
                    self.dones[agent_name] = False
                    self.rewards[agent_name] = 0
                    self._cumulative_rewards[agent_name] = 0
                    self.infos[agent_name] = {}
                    self.id_nr += 1
                case self.grass_type_nr:
                    grass = Grass(x, y, self.id_nr)
                    type_name = grass.agent_type_name
                    agent_name = f"{type_name}_{self.id_nr}"
                    grass.agent_name = agent_name
                    grass.energy_level = initial_energy_level_grass
                    grass.age = 0
                    # localization helpers
                    self.n_agents_in_grid_cells[self.grass_type_nr][x, y] += 1
                    if self.n_agents_in_grid_cells[self.grass_type_nr][x, y] > 1:
                        print("ERROR: NOT MORE THAN ONE GRASS AGENT PER CELL")
                    self.agents_lists_in_grid_cells[x][y][self.grass_type_nr].append(grass)
                    self.agents_instance_list.append(grass)
                    self.agents.append(agent_name)
                    self.agents_dict[agent_name] = grass
                    self.n_active_agents[self.grass_type_nr] += 1
                    self.dones[agent_name] = False
                    self.rewards[agent_name] = 0
                    self._cumulative_rewards[agent_name] = 0
                    self.infos[agent_name] = {}
                    self.id_nr += 1
            return agent_name

        def seed(self, seed=None):
            pass

        def get_random_location(self):
            # TODO: research implementation seed
            x = random.randint(0, self.x_size-1)
            y = random.randint(0, self.x_size-1)
            print("(x,y)= ("+str(x)+","+str(y)+")")
            return x, y

        def get_neighborhood_agents(self, agent):
            """
            Returns the number of the different agents for each neighbor cell
            """
            # current_location = agent.location
            x = env.agents_dict[agent].x
            y = env.agents_dict[agent].y

            n_agents_in_neighborhood = np.zeros([self.n_agent_types, self.n_neighborhood_cells])
            moore_index = 0
            for d_x in self.relative_x_positions_neighbors:
                for d_y in self.relative_y_positions_neighbors:
                    x_neighbor_location = (x + d_x) % x_size
                    y_neighbor_location = (y + d_y) % x_size
                    n_agents_in_neighborhood[self.grass_type_nr][moore_index] = \
                        len(self.agents_lists_in_grid_cells[x_neighbor_location][y_neighbor_location][
                                self.grass_type_nr])
                    n_agents_in_neighborhood[self.prey_type_nr][moore_index] = \
                        len(self.agents_lists_in_grid_cells[x_neighbor_location][y_neighbor_location][
                                self.prey_type_nr])

                    n_agents_in_neighborhood[self.predator_type_nr][moore_index] = \
                        len(self.agents_lists_in_grid_cells[x_neighbor_location][y_neighbor_location][
                                self.predator_type_nr])

                    moore_index += 1
            return n_agents_in_neighborhood

        def get_empty_grass_location(self):
            grass_instance = self.instance(self.agent_selection)
            x_grass = grass_instance.x
            y_grass = grass_instance.y
            possible_grass_offspring_moore_index_list = []
            moore_index = 0
            for d_x in self.relative_x_positions_neighbors:
                for d_y in self.relative_y_positions_neighbors:
                    x_neighbor_location = (x_grass + d_x) % x_size
                    y_neighbor_location = (y_grass + d_y) % x_size
                    if not self.is_cell_occupied_with_grass(x_neighbor_location, y_neighbor_location):
                        possible_grass_offspring_moore_index_list.append(moore_index)
                    moore_index += 1

            if possible_grass_offspring_moore_index_list:
                random_moore_index = random.randint(0, len(possible_grass_offspring_moore_index_list)-1)
                random_moore_location = possible_grass_offspring_moore_index_list[random_moore_index]
                x = (grass_instance.x + self.actions_positions_dict[random_moore_location][0]) % self.x_size
                y = (grass_instance.y + self.actions_positions_dict[random_moore_location][1]) % self.x_size
                return x, y
            else:
                return None

        def get_features(self, agent):
            """
            Returns the features for a position (x,y) as a matrix 9x3
            Row 1: predators, Row 2: prey, Row 3: grass
            """
            features = np.zeros(self.n_features)
            n_agents_in_neighborhood = self.get_neighborhood_agents(agent)
            if self.n_active_agents[self.prey_type_nr] != 0:
                features[0] = sum(n_agents_in_neighborhood[self.prey_type_nr][:]) / self.n_active_agents[
                    self.prey_type_nr]
            else:
                features[0] = 0
            if self.n_active_agents[self.predator_type_nr] != 0:
                features[1] = sum(n_agents_in_neighborhood[self.predator_type_nr][:]) / \
                              self.n_active_agents[self.predator_type_nr]
            else:
                features[1] = 0
            if self.n_active_agents[self.grass_type_nr] != 0:
                features[2] = sum(n_agents_in_neighborhood[self.grass_type_nr][:]) / self.n_active_agents[
                    self.grass_type_nr]
            else:
                features[2] = 0
            n_prey_in_neighborhood = sum(n_agents_in_neighborhood[self.prey_type_nr][:])
            if n_prey_in_neighborhood == 0:
                features[3:self.n_features] = 0
            else:
                for f in range(self.n_neighborhood_cells):
                    features[f + 3] = n_agents_in_neighborhood[self.prey_type_nr][f] / n_prey_in_neighborhood
            return features

        def get_reward(self, agent):
            """
            reward = opponent * type + 2 * same * type
            opponent = number of the other species type within the agent’s Moore
            neighborhood normalized by the number of that species in the world
            type is 1 for predator and −1 for prey
            same = {0, 1} for if the opponent is on the same location

            """
            if env.agents_dict[agent].agent_type_name == "predator":
                type_animal = 1
                n_agents_in_neighborhood = self.get_neighborhood_agents(agent)
                if self.n_active_agents[self.prey_type_nr] != 0:
                    relative_prey = sum(n_agents_in_neighborhood[self.prey_type_nr][
                                        :]) / self.n_active_agents[self.prey_type_nr]

                else:
                    relative_prey = 0
                opponent = relative_prey
                center_position = int((self.n_neighborhood_cells - 1) / 2)
                # also prey in center position
                same = n_agents_in_neighborhood[self.prey_type_nr][center_position] > 0
                reward = opponent * type_animal + 2 * same * type_animal

            elif env.agents_dict[agent].agent_type_name == "prey":
                type_animal = -1
                n_agents_in_neighborhood = self.get_neighborhood_agents(agent)
                if self.n_active_agents[self.predator_type_nr] != 0:
                    relative_predators = sum(n_agents_in_neighborhood[self.predator_type_nr][
                                             :]) / self.n_active_agents[self.predator_type_nr]
                else:
                    relative_predators = 0
                opponent = relative_predators
                center_position = int((self.n_neighborhood_cells - 1) / 2)
                # also predator(s) in same (center) position as prey
                same = n_agents_in_neighborhood[self.predator_type_nr][center_position] > 0
                reward = opponent * type_animal + 2 * same * type_animal
            else:  # predator nor prey i.e. grass
                reward = 0
            return reward

        def observe(self, agent_name):
            observation = self.n_agents_in_grid_cells
            return observation

        def reset(self, seed=None):
            #if seed is not None:
            #    self.seed(seed=seed)
            self.observations = {agent: None for agent in self.agents}
            for j in range(self.n_initial_predators):
                x, y = self.get_random_location()
                self.create_agent(self.predator_type_nr, x, y)
            for j in range(self.n_initial_prey):
                x, y = self.get_random_location()
                self.create_agent(self.prey_type_nr, x, y)
            for j in range(self.n_initial_grass):
                x, y = self.get_random_location()
                while self.is_cell_occupied_with_grass(x,y):
                    x, y = self.get_random_location()
                self.create_agent(self.grass_type_nr, x, y)
            self._agent_selector = agent_selector(self.agents)
            self.agent_selection = self._agent_selector.next()

        def last(self, observe=True):
            agent_name = self.agent_selection
            done = self.dones[agent_name]
            info = self.infos[agent_name]
            reward = self.get_reward(agent_name)
            observation = self.observe(agent_name)
            return observation, reward, done, info

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
                    prey_count = np.transpose(observation[self.prey_type_nr])[j][i]
                    count_text: str
                    if prey_count < 1:
                        count_text = ""
                    elif prey_count < 10:
                        count_text = str(prey_count)
                    else:
                        count_text = "+"
                    text = font.render(count_text, False, prey_color)
                    self.screen.blit(text, (pos_x, pos_y))
                    predator_count = np.transpose(observation[self.predator_type_nr])[j][i]
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
            for agent in self.agents_instance_list:
                scale = 1
                if agent.agent_name == self.agent_selection:
                    scale = 2
                x = agent.x
                y = agent.y
                if agent.agent_type_name == "predator":

                    center = (int(self.pixel_scale * x + self.pixel_scale / 2),
                              int(self.pixel_scale * y + self.pixel_scale / 2))
                    pygame.draw.circle(self.screen, predator_color, center,
                                       int(self.pixel_scale*scale / 4))
                    # id_nr
                    text = font.render(str(agent.id_nr), False, (255, 255, 225))
                    size_id_nr = len(str(agent.id_nr))
                    center_id = (int(self.pixel_scale * x + self.pixel_scale / (2+(size_id_nr*5)/10)),
                                 int(self.pixel_scale * y + self.pixel_scale / 2.5))
                    self.screen.blit(text, center_id)

                elif agent.agent_type_name == "prey":

                    center = (int(self.pixel_scale * x + self.pixel_scale / 2),
                              int(self.pixel_scale * y + self.pixel_scale / 2))
                    pygame.draw.circle(self.screen, prey_color, center,
                                       int(self.pixel_scale*scale / 4))
                    #id_nr
                    text = font.render(str(agent.id_nr), False, (255, 255, 225))
                    size_id_nr = len(str(agent.id_nr))
                    center_id = (int(self.pixel_scale * x + self.pixel_scale / (2+(size_id_nr*5)/10)),
                                 int(self.pixel_scale * y + self.pixel_scale / 2.5))
                    self.screen.blit(text, center_id)
                elif agent.agent_type_name == "grass":
                    # color='lightgreen'
                    pos = pygame.Rect(self.pixel_scale * x + 6, self.pixel_scale * y + 6,
                                      self.pixel_scale - 12, self.pixel_scale - 12)
                    pygame.draw.rect(self.screen, grass_color, pos, 10*scale)
                    # id_nr
                    text = font.render(str(agent.id_nr), False, grass_color)
                    size_id_nr = len(str(agent.id_nr))
                    center_id = (int(self.pixel_scale * x + self.pixel_scale / (6+(size_id_nr*5)/10)),
                                 int(self.pixel_scale * y + self.pixel_scale / 6))
                    self.screen.blit(text, center_id)


            pygame.display.update()

        def get_random_agent_grid_cell(self, agent_type_nr, x, y):
            agent_instance = None
            match agent_type_nr:
                case self.prey_type_nr:
                    prey_list = self.agents_lists_in_grid_cells[x][y][self.prey_type_nr]
                    # prey available at new predator location?
                    if prey_list:
                        random_prey_index = random.randint(0, len(prey_list)-1)
                        agent_instance = prey_list[random_prey_index]
                case self.grass_type_nr:
                    grass_list = self.agents_lists_in_grid_cells[x][y][self.grass_type_nr]
                    # grass available at new prey location?
                    if grass_list:
                        agent_instance = grass_list[0]
            return agent_instance

        def step(self, action):
            if self.dones[self.agent_selection]:
                self._was_done_step(action)
                return
            # main contents of step
            agent_instance = self.instance(self.agent_selection)
            match agent_instance.agent_type_nr:
                case self.predator_type_nr:
                    predator = agent_instance
                    predator.age += 1
                    predator.energy_level -= self.predators_metabolism_energy
                    predator.energy_level -= self.execute_action_selected_agent(action)
                    if predator.energy_level <= 0:
                        self.dones[predator.agent_name] = True
                        print(predator.agent_name + " has no energy left and is done")
                    elif predator.energy_level <= self.predators_reproduction_energy_minimum:
                        eaten_prey_instance = self.get_random_agent_grid_cell(self.prey_type_nr, predator.x, predator.y)
                        if eaten_prey_instance:
                            self.dones[eaten_prey_instance.agent_name] = True
                            predator.energy_level += eaten_prey_instance.energy_level
                            print(predator.agent_name + " is eating " + eaten_prey_instance.agent_name +
                                  " and has energy-level "+str(predator.energy_level))
                    elif predator.energy_level > self.predators_reproduction_energy_minimum:
                        # energy level above "predators_reproduction_energy_minimum"
                        predator.energy_level -= self.predators_cost_of_reproduction
                        #new_predator_name = self.create_agent(self.predator_type_nr, predator.x, predator.y)
                        #print(new_predator_name+" created from "+ predator.agent_name)

                case self.prey_type_nr:
                    prey = agent_instance
                    prey.age += 1
                    prey.energy_level -= self.prey_metabolism_energy
                    prey.energy_level -= self.execute_action_selected_agent(action)
                    if prey.energy_level <= 0:
                        self.dones[prey.agent_name] = True
                        print(prey.agent_name + " has no energy left and is done")
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
                                print(prey.agent_name + " is eating " + eaten_grass_instance.agent_name +
                                      " and has energy-level " + str(prey.energy_level) + ", grass dies ")
                            else:
                                print(prey.agent_name + " is eating " + eaten_grass_instance.agent_name +
                                      " and has energy-level " + str(prey.energy_level) + ", grass stays alive")
                    elif prey.energy_level > self.prey_reproduction_energy_minimum:
                        prey.energy_level -= self.prey_cost_of_reproduction
                        #new_prey_name = self.create_agent(self.prey_type_nr, prey.x, prey.y)
                        #print(new_prey_name+" created from " + prey.agent_name)

                case self.grass_type_nr:
                    grass = agent_instance
                    grass.energy_level += self.grass_growth
                    if grass.energy_level > self.grass_reproduction_energy_minimum \
                            and not self.get_empty_grass_location() == None:
                        x, y = self.get_empty_grass_location()
                        new_grass_name = self.create_agent(self.grass_type_nr, x, y)
                        print(new_grass_name+" created @ ["+str(x)+","+str(y)+"] from " + grass.agent_name+ " @ ["+str(grass.x)+","+str(grass.y)+"]")

            self.agent_selection = self._agent_selector.next()
            self.agent_selection = self._dones_step_first()

        def _was_done_step(self, action):
            """
            Helper function that performs step() for done agents.
            """
            if action is not None:
                raise ValueError("when an agent is done, the only valid action is None")
            """
            the sequence of living agents can give rise to unexpected behavior 
            in the following cases of "done" and subsequent removal:

                1) the selected agent causes another agent to "done" where the other agent has 
                already been selected earlier in the agent_order of the same cycle,
                2) the selected agent causes itself to "done"
                3) the selected agents causes its direct successor in the agent_order to "done".
            """
            # to fix a runtime error when self._skip_agent_selection is already being "done"
            # and deleted by the selected agent: see 3)
            if self.agent_selection == self._skip_agent_selection:
                self._skip_agent_selection = self._agent_selector.next()

            # removes done agent
            agent_name = self.agent_selection
            assert self.dones[
                agent_name
            ], "an agent that was not done as attempted to be removed"
            print("FRONT LOADING")
            self.remove_agent(agent_name)

            # finds next done agent or loads next live agent (Stored in _skip_agent_selection)
            _dones_order = [agent_name for agent_name in self.agents if self.dones[agent_name]]
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


    def policy(observation, agent):
        # observation_space = env.observation_spaces(agent)
        action = None
        if env.agents_dict[agent].agent_type_name == "predator":
            action = random.randint(0,8) #env.act_space.sample()

        elif env.agents_dict[agent].agent_type_name == "prey":
            action = random.randint(0,8) #env.act_space.sample()

        elif env.agents_dict[agent].agent_type_name == "grass":
            action = 4
        return action


    env = PredatorPreyGridEnv(x_size,
                              n_initial_predators,
                              n_initial_prey,
                              n_initial_grass)

    env.reset()
    iter = 0
    cycle = 0
    for agent in env.agent_iter(env.n_iterations):
        if env._agent_selector.is_first():
            cycle += 1
            print("cycle "+str(cycle))
        print("agent_selection "+env.agent_selection+" energy: "+str(env.instance(env.agent_selection).energy_level))
        print(env.agents)
        observation, reward, done, info = env.last()
        action = policy(observation, agent) if not done else None
        env.step(action)
        env.render()
        time.sleep(sleep_time)
        print()
        iter += 1

    input('Press any key to exit\n')
