import numpy as np
import json

class UniformModel:
    def get_probabilities(self, _):
        return np.ones(4) / 4

class Model:
    def __init__(self):
        self._table = {}
        self._increment = 10
        self._weights_file = "model.json"

    def get_probabilities(self, pattern):
        if pattern not in self._table:
            self._table[pattern] = np.ones(4)

        return self._table[pattern] / np.sum(self._table[pattern])

    def search_pattern(self, pattern):
        if pattern not in self._table:
            self._table[pattern] = np.ones(4)

        return self._table[pattern]
    
    def update(self, path):
        states = path.get_states()
        actions = path.get_actions()
        for i in range(path.length()):
            pattern = states[i].get_pattern()
            action = actions[i]
            self.search_pattern(pattern)[action] += self._increment

    def save_weight(self):
        """
        Save model weights
        """
        data = {json.dumps([float(x) for x in k]): v.tolist()
                for k, v in self._table.items()}
        with open(self._weights_file, "w") as f:
            json.dump(data, f)

    def load_weight(self):
        with open(self._weights_file) as f:
            data = json.load(f)
        self._table = {tuple(json.loads(k)): np.array(v) for k, v in data.items()}

class OptimizedModel(Model):
    def __init__(self):
        super().__init__()
        self._weights_file = "optimized_model.json"

    def update(self, path):
        states = path.get_states()
        actions = path.get_actions()
        for i in range(path.length()):
            regular, reversed = states[i].get_reversed_pattern()
            action = actions[i]
            self.search_pattern(regular)[action] += self._increment
            if reversed != regular:
                self.search_pattern(reversed)[action] += self._increment