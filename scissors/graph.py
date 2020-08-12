import numpy as np
from scissors.search import search
import time


class PathFinder:
    def __init__(self, cost, maximum_cost):
        self.cost = cost.astype(np.int)
        self.maximum_cost = maximum_cost

    def find_path(self, seed_x, seed_y,  free_x, free_y, dynamic_cost):
        h, w = self.cost.shape[2:]

        if dynamic_cost is None:
            dynamic_cost = np.zeros((h, w), dtype=np.int)

        start = time.time()
        node_map = search(self.cost, dynamic_cost, w, h, seed_x, seed_y, self.maximum_cost)

        cur_x, cur_y = node_map[:, free_x, free_y]
        history = []

        while (cur_x, cur_y) != (seed_x, seed_y):
            history.append((cur_y, cur_x))
            cur_x, cur_y =  node_map[:, cur_x, cur_y]

        # print(time.time() - start)
        return history
