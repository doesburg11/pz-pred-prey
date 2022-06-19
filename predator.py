class Predator:
    agent_type_nr = 0
    agent_type_name = "predator"

    def __init__(self, x, y, id_nr):

        self.x = x
        self.y = y
        self.id_nr = id_nr

        self.agent_name = str()
        self.energy_level = 0
        self.age = 0
        self.weights = []
