import numpy as np
from .search import search
import time


class PathFinder:
    def __init__(self, cost, maximum_cost):
        self.cost = cost.astype(np.int)
        self.maximum_cost = maximum_cost

    def find_path(self, seed_point, free_point, abc):
        h, w = self.cost.shape[2:]

        seed_x = seed_point[1]
        seed_y = seed_point[0]

        free_x = free_point[1]
        free_y = free_point[0]

        start = time.time()
        history = search(self.cost, w, h, seed_x, seed_y, free_x, free_y, maximum_local_cost=self.maximum_cost)

        print(time.time() - start)
        return history
