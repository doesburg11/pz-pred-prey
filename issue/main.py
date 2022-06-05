from pettingzoo import AECEnv

class agent_selector:
    """
    Outputs an agent in the given order whenever agent_select is called. Can reinitialize to a new order
    """

    def __init__(self, agent_order):
        self.reinit(agent_order)

    def reinit(self, agent_order):
        self.agent_order = agent_order
        self._current_agent = 0
        self.selected_agent = 0

    def reset(self):
        self.reinit(self.agent_order)
        return self.next()

    def next(self):
        self._current_agent = (self._current_agent + 1) % len(self.agent_order)
        self.selected_agent = self.agent_order[self._current_agent - 1]
        return self.selected_agent

    def is_last(self):
        """
        Does not work as expected if you change the order
        """
        return self.selected_agent == self.agent_order[-1]

    def is_first(self):
        return self.selected_agent == self.agent_order[0]

    def __eq__(self, other):
        if not isinstance(other, agent_selector):
            return NotImplemented

        return (
            self.agent_order == other.agent_order
            and self._current_agent == other._current_agent
            and self.selected_agent == other.selected_agent
        )



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

        def _was_done_step(self, action):
            """
            Helper function that performs step() for done agents.

            Does the following:

            1. Removes done agent from .agents, .dones, .rewards, ._cumulative_rewards, and .infos
            2. Loads next agent into .agent_selection: if another agent is done, loads that one, otherwise load next live agent
            3. Clear the rewards dict

            Highly recommended to use at the beginning of step as follows:

            def step(self, action):
                if self.dones[self.agent_selection]:
                    self._was_done_step()
                    return
                # main contents of step
            """
            if action is not None:
                raise ValueError("when an agent is done, the only valid action is None")

            # removes done agent
            agent = self.agent_selection
            assert self.dones[
                agent
            ], "an agent that was not done as attempted to be removed"
            print("to be removed agent "+agent+" has array index: "+str(self.agents.index(agent))+" (='_current_agent' number minus 1)")
            print("'_skip_agent_selection': "+self._skip_agent_selection +" has '_current_agent' number: "+str( self._agent_selector._current_agent)+" and array index "+str(self.agents.index("agent_3")))
            del self.dones[agent]
            del self.rewards[agent]
            del self._cumulative_rewards[agent]
            del self.infos[agent]
            self.agents.remove(agent)
            print("agent "+agent+" is removed from agents")
            print("'_skip_agent_selection': "+self._skip_agent_selection +" has '_current_agent' number: "+str( self._agent_selector._current_agent)+" and array index "+str(self.agents.index("agent_3")))

            if self.agents[self._agent_selector._current_agent-1] != self.agent_selection :
                self._agent_selector._current_agent -= 1

            # finds next done agent or loads next live agent (Stored in _skip_agent_selection)
            _dones_order = [agent for agent in self.agents if self.dones[agent]]
            if _dones_order:
                if getattr(self, "_skip_agent_selection", None) is None:
                    self._skip_agent_selection = self.agent_selection
                self.agent_selection = _dones_order[0]
            else:
                if getattr(self, "_skip_agent_selection", None) is not None:
                    self.agent_selection = self._skip_agent_selection
                self._skip_agent_selection = None
            self._clear_rewards()

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
