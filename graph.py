from dijkstar import Graph, find_path


class PathFinder:
    def __init__(self, shape, cost):
        self.graph = self.create_graph(shape, cost)

    def find_path(self, start, stop):
        start_h, start_w = start
        stop_h, stop_w = stop
        path = find_path(self.graph, self.get_edge_id(start_h, start_w), self.get_edge_id(stop_h, stop_w))

        coord = []
        for node in path.nodes:
            coord.append(self.get_coords(node))
        return coord

    # TODO do better
    @staticmethod
    def get_edge_id(h, w):
        return f'{h}.{w}'

    @staticmethod
    def get_coords(node):
        return node.split('.')

    @staticmethod
    def create_graph(shape, cost):
        h, w = shape
        graph = Graph()

        cost = IndexBasedGraphWrapper(cost)
        get_id = PathFinder.get_edge_id

        # TODO looks very bad (((
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
