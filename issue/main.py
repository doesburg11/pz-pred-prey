from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

if __name__ == '__main__':

    class Agent:
        def __init__(self, id_nr):
            self.id_nr = id_nr

            self.agent_name = None
            self.energy_level = 0

    class test_env(AECEnv):
        metadata = {'render.modes': ['human']}

        def __init__(self):

            super().__init__()

            self.observations = dict()
            self._cumulative_rewards = dict()
            self.rewards = dict()
            self.infos = dict()
            self.dones = dict()
            self.agents_dict = dict()  # links "agent" (agents_name) to the actual agent_instance
            self._agent_selector = None
            self.agents = []  # agent_0, ..., agent_n-1, ... ,agent_n]
            self.id_nr = 0

        def instance(self, agent_name):  # returns the agent_instance
            return self.agents_dict[agent_name]

        def create_agent(self):
            agent_instance = Agent(self.id_nr)
            type_name = "agent"
            agent_name = f"{type_name}_{self.id_nr}"
            agent_instance.agent_name = agent_name
            if self.id_nr == 2:  # for illustration; only agent_instance_2 runs out of energy
                agent_instance.energy_level = 2
            else:
                agent_instance.energy_level = 100
            self.agents_dict[agent_name] = agent_instance
            self.agents.append(agent_name)
            self.dones[agent_name] = False
            self.rewards[agent_name] = 0
            self._cumulative_rewards[agent_name] = 0
            self.infos[agent_name] = {}
            self.observations[agent_name] = 0
            self.id_nr += 1

        def reset(self, seed=None):
            for _ in range(6):
                self.create_agent()
            self._agent_selector = agent_selector(self.agents)
            self.agent_selection = self._agent_selector.next()

        def last(self, observe=True):
            agent_name = self.agent_selection
            done = self.dones[agent_name]
            info = {}
            reward = 0
            observation = 0
            return observation, reward, done, info

        def render(self, mode="human"):
            print("iteration " + str(iter) + ": agent: " + str(agent))

        def step(self, action):
            if self.dones[self.agent_selection]:
                self._was_done_step(action)
                return
            agent_name = self.agent_selection
            agent_instance = self.instance(agent_name)
            agent_instance.energy_level -= 1
            if agent_instance.energy_level <= 0:
                self.dones[agent_name] = True
                print(agent_name + " HAS NO ENERGY LEFT AND IS DONE")

            if self._agent_selector.is_last():
                self._agent_selector.reinit(self.agents)

            self.agent_selection = self._agent_selector.next()
            self.agent_selection = self._dones_step_first()

    def policy(observation, agent_name):
        # observation_space = env.observation_spaces(agent)
        action = 0
        return action

    env = test_env()

    env.reset()
    iter = 0
    for agent in env.agent_iter(25):
        env.render()
        observation, reward, done, info = env.last()
        action = policy(observation, agent) if not done else None
        env.step(action)
        iter += 1
