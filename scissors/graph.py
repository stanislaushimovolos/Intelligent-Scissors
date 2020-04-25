import numpy as np
from dijkstar import Graph, find_path


class PathFinder:
    def __init__(self, shape, cost):
        self.shape = shape
        self.map_value = None
        self.graph = self.create_graph(shape, cost)

    def find_path(self, start, stop, cost_func=None):
        if cost_func is None:
            path = find_path(
                self.graph, self.get_node_from_pos(*start),
                self.get_node_from_pos(*stop)
            )
        else:
            path = find_path(
                self.graph, self.get_node_from_pos(*start),
                self.get_node_from_pos(*stop),
                cost_func=cost_func
            )
        coord = []
        for node in path.nodes:
            coord.append(self.get_pos_from_node(node))
        return coord

    def get_node_from_pos(self, h, w):
        return self.map_value * h + w

    def get_pos_from_node(self, node_id):
        return node_id // self.map_value, node_id % self.map_value

    def create_graph(self, shape, cost):
        h, w = shape
        graph = Graph()

        map_value = np.max(shape) + 1
        self.map_value = map_value
        get_id = self.get_node_from_pos

        cost = IndexBasedGraphWrapper(cost)
        for i in range(1, w - 1):
            for j in range(1, h - 1):
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
