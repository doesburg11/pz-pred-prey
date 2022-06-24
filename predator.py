import numpy as np


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
        self.weights =np.zeros(12)
        self.weights_array = []
        self.is_learning = True
        self.q = 0

    def get_q_function(self, features):
        weights = self.weights
        q = 0
        for i in range(len(weights)):
            q = q + weights[i] * features[i]
        return q

    def update_weight(self, env, reward, q_value):
        weights = self.weights
        learning_rate = env.learning_rate
        discount_factor = env.discount_factor

        # Compute the Q'-table:
        q_prime = []
        for d_x in [-1, 0, 1]:
            for d_y in [-1, 0, 1]:
                x_target = (self.x + d_x) % env.x_size
                y_target = (self.y + d_y) % env.x_size
                features = env.get_features(self.agent_name, x_target, y_target)
                q_prime.append(self.get_q_function(features))

        # Update the weights:
        q_prime_max = max(q_prime)
        for i in range(0, len(weights)):
            if i < 3:
                c = 9 / (env.x_size * env.x_size)
            else:
                c = 1 / 9

            w = weights[i]
            f = features[i]
            f = np.exp(-0.5 * (f - c) ** 2)

            weights[i] = w + learning_rate * (reward + discount_factor * q_prime_max - q_value) * f

        self.weights = weights
        return
