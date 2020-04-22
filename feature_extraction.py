import numpy as np
from dijkstar import Graph
from itertools import product
from skimage.filters import laplace, sobel_h, sobel_v

from utils import unfold

shifts = [(-1, 1), (-1, 0), (-1, -1), (0, 1), (0, -1), (1, -1), (1, 0), (1, 1)]

default_weights = {
    'laplace': 0,
    'direction': 0.9,
    'magnitude': 0.1
}


class StaticFeatureExtractor:
    def __init__(self, laplace_kernel_size=5, weights=None, connect_area=3):
        if weights is None:
            weights = default_weights

        self.laplace_w = weights['laplace']
        self.magnitude_w = weights['magnitude']
        self.direction_w = weights['direction']

        self.laplace_kernel_size = laplace_kernel_size
        self.filter_size = np.array([connect_area, connect_area])

    def get_total_link_costs(self, image):
        l_cost = self.get_laplace_cost(image, self.laplace_kernel_size)
        l_cost, _ = unfold(l_cost, self.filter_size)

        g_cost = self.get_gradient_magnitude_cost(image)
        g_cost, _ = unfold(g_cost, self.filter_size)

        d_cost = self.get_gradient_direction_feats(image)
        total_cost = self.laplace_w * l_cost + self.magnitude_w * g_cost + self.direction_w * d_cost
        return total_cost

    @staticmethod
    def get_laplace_cost(img, kernel_size):
        cost = laplace(img, ksize=kernel_size)
        cost = np.expand_dims(cost, 0)
        return cost < 0.1

    @staticmethod
    def get_gradient_magnitude_cost(image):
        grads = np.array([sobel_h(image), sobel_v(image)])
        grads = np.transpose(grads, (1, 2, 0))
        norm = np.linalg.norm(grads, axis=2)
        cost = 1 - norm / np.max(norm)
        cost = np.expand_dims(cost, 0)
        return cost

    @staticmethod
    def create_spatial_feats(shape, filter_size):
        start_span_coord = filter_size // 2
        stop_span_coord = filter_size - start_span_coord - 1
        shift_boundaries = [
            np.arange(-start_coord, stop_coord + 1)
            for start_coord, stop_coord in zip(start_span_coord, stop_span_coord)
        ]

        feats = np.zeros(((2,) + (3, 3) + shape))
        for shift in product(*shift_boundaries):
            current_slice = tuple(shift + start_span_coord)
            feats[:, current_slice[0], current_slice[1]] = np.reshape(shift, (2, 1, 1))
            if shift != (0, 0):
                feats[:, current_slice[0], current_slice[1]] /= np.linalg.norm(shift)
        return feats

    def get_gradient_direction_feats(self, img):
        grads = np.stack([-sobel_h(img), sobel_v(img)])
        unfolded_grads, _ = unfold(grads, self.filter_size)
        grads = grads[:, np.newaxis, np.newaxis, ...]

        spatial_feats = self.create_spatial_feats(img.shape, self.filter_size)
        tmp = np.einsum('i..., i...', spatial_feats, grads)

        sign_mask = np.sign(tmp)
        local_feats = sign_mask * tmp
        distant_feats = sign_mask * np.einsum('i..., i...', spatial_feats, unfolded_grads)

        total_cost = 2 / (3 * np.pi) * (np.arccos(local_feats) + np.arcsin(distant_feats))
        return total_cost


class DynamicFeatureExtractor:
    def __init__(self):
        pass


# TODO remove this trash
def create_graph(shape, cost):
    graph = Graph()
    w, h = shape
    for i in range(h):
        for j in range(w):
            if j != 0:
                graph.add_edge(f'{i}.{j}', f'{i}.{j - 1}', cost[1, 0, i, j])
            if j != w - 1:
                graph.add_edge(f'{i}.{j}', f'{i}.{j + 1}', cost[1, 1, i, j])
            if i != 0:
                graph.add_edge(f'{i}.{j}', f'{i - 1}.{j}', cost[0, 1, i, j])

            if i != h - 1:
                graph.add_edge(f'{i}.{j}', f'{i + 1}.{j}', cost[2, 1, i, j])

    return graph
