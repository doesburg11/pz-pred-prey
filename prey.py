class Prey:
    agent_type_nr = 1
    agent_type_name = "prey"

    def __init__(self,
                 x,
                 y,
                 id_nr):

        self.x = x
        self.y = y
        self.id_nr = id_nr

        self.agent_name = "prey_" + str(self.id_nr)
        self.energy_level = 0
        self.age = 0

