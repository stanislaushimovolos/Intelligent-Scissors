import numpy as np
from functools import partial
from dijkstar import Graph, find_path

from scissors.utils import get_static_cost, get_pos_from_node, get_node_from_pos


class PathFinder:
    def __init__(self, shape, cost):
        self.shape = shape
        self.index_key = np.max(shape) + 2
        self.graph = self.create_graph(shape, cost)

    def find_path(self, start, stop, cost_func=get_static_cost):
        path = find_path(
            self.graph, get_node_from_pos(*start, self.index_key),
            get_node_from_pos(*stop, self.index_key),
            cost_func=cost_func
        )
        coord = []
        for node in path.nodes:
            coord.append(get_pos_from_node(node, self.index_key))
        return coord

    def create_graph(self, shape, cost):
        w, h = shape
        graph = Graph()
        get_id = partial(get_node_from_pos, index_key=self.index_key)

        cost = IndexBasedGraphWrapper(cost)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                graph.add_edge(get_id(i, j), get_id(i, j - 1), cost.left[i, j])
                graph.add_edge(get_id(i, j), get_id(i, j + 1), cost.right[i, j])
                graph.add_edge(get_id(i, j), get_id(i - 1, j), cost.top[i, j])
                graph.add_edge(get_id(i, j), get_id(i + 1, j), cost.bottom[i, j])
                graph.add_edge(get_id(i, j), get_id(i + 1, j + 1), cost.right_bottom[i, j])
                graph.add_edge(get_id(i, j), get_id(i - 1, j - 1), cost.left_top[i, j])
                graph.add_edge(get_id(i, j), get_id(i + 1, j - 1), cost.left_bottom[i, j])
                graph.add_edge(get_id(i, j), get_id(i - 1, j + 1), cost.right_top[i, j])
        return graph


class IndexBasedGraphWrapper:
    def __init__(self, index_graph):
        self.index_graph = index_graph

    @property
    def bottom(self):
        return self.index_graph[2, 1, :, :]

    @property
    def top(self):
        return self.index_graph[0, 1, :, :]

    @property
    def right(self):
        return self.index_graph[1, 2, :, :]

    @property
    def right_top(self):
        return self.index_graph[0, 2, :, :]

    @property
    def right_bottom(self):
        return self.index_graph[2, 2, :, :]

    @property
    def left(self):
        return self.index_graph[1, 0, :, :]

    @property
    def left_top(self):
        return self.index_graph[0, 0, :, :]

    @property
    def left_bottom(self):
        return self.index_graph[2, 0, :, :]
