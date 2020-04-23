import numpy as np
from dijkstar import Graph
from itertools import product
from skimage.filters import gaussian, laplace, sobel_h, sobel_v

from utils import unfold

shifts = [(-1, 1), (-1, 0), (-1, -1), (0, 1), (0, -1), (1, -1), (1, 0), (1, 1)]

default_weights = {
    'laplace': 0.2,
    'direction': 0.7,
    'magnitude': 0.1
}


class StaticFeatureExtractor:
    def __init__(self, gaussian_std=2, laplace_filter_size=5, weights=None, connected_area=3):
        if weights is None:
            weights = default_weights

        self.laplace_w = weights['laplace']
        self.magnitude_w = weights['magnitude']
        self.direction_w = weights['direction']

        self.gaussian_std = gaussian_std
        self.laplace_filter_size = laplace_filter_size
        self.filter_size = np.array([connected_area, connected_area])

    def get_total_link_costs(self, image):
        l_cost = self.get_laplace_cost(image, self.laplace_filter_size, self.gaussian_std)
        l_cost = unfold(l_cost, self.filter_size)

        g_cost = self.get_magnitude_cost(image)
        g_cost = unfold(g_cost, self.filter_size)

        d_cost = self.get_direction_cost(image)
        total_cost = self.laplace_w * l_cost + self.magnitude_w * g_cost + self.direction_w * d_cost
        return total_cost

    @staticmethod
    def get_laplace_cost(image, filter_size, std):
        output = np.ones_like(image)
        blurred = gaussian(image, std)
        laplace_map = laplace(blurred, ksize=filter_size)

        unfolded_laplace = unfold(
            np.expand_dims(laplace_map, 0),
            np.array([filter_size, filter_size])
        )
        unfolded_laplace = np.reshape(
            unfolded_laplace, (filter_size * filter_size,) + laplace_map.shape
        )
        n_positive = np.sum(unfolded_laplace >= 0, axis=0)
        n_negative = np.sum(unfolded_laplace < 0, axis=0)

        # TODO check minimum value
        zero_crossing_mask = ((n_negative > 0) & (n_positive > 0))
        output[zero_crossing_mask] = 0
        return np.expand_dims(output, 0)

    @staticmethod
    def get_magnitude_cost(image):
        grads = np.array([sobel_h(image), sobel_v(image)])
        grads = np.transpose(grads, (1, 2, 0))
        norm = np.linalg.norm(grads, axis=2)
        cost = 1 - norm / np.max(norm)
        cost = np.expand_dims(cost, 0)
        return cost

    @staticmethod
    def create_spatial_feats(shape, filter_size, feature_size=2):
        start_span_coord = filter_size // 2
        stop_span_coord = filter_size - start_span_coord - 1
        shift_boundaries = [
            np.arange(-start_coord, stop_coord + 1)
            for start_coord, stop_coord in zip(start_span_coord, stop_span_coord)
        ]

        holder = np.zeros((feature_size,) + tuple(filter_size) + shape)
        for shift in product(*shift_boundaries):
            current_slice = shift + start_span_coord
            shift = np.reshape(shift, (feature_size,) + (1,) * 2 * len(filter_size))

            slices = (slice(None),) + tuple([slice(x, x + 1) for x in current_slice])
            holder[slices] = shift
            if shift.any():
                holder[slices] /= np.linalg.norm(shift)

        return holder

    def get_direction_cost(self, img):
        grads = np.stack([-sobel_h(img), sobel_v(img)])
        unfolded_grads = unfold(grads, self.filter_size)
        grads = grads[:, None, None, ...]

        spatial_feats = self.create_spatial_feats(img.shape, self.filter_size)
        link_feats = np.einsum('i..., i...', spatial_feats, grads)
        local_feats = np.abs(link_feats)

        sign_mask = np.sign(link_feats)
        distant_feats = sign_mask * np.einsum('i..., i...', spatial_feats, unfolded_grads)
        total_cost = 2 / (3 * np.pi) * (np.arccos(local_feats) + np.arcsin(distant_feats))
        return total_cost


class PixelEdgesWrapper:
    def __init__(self, pixel_neighbours):
        self.neighbours = pixel_neighbours


class IndexBasedGraphWrapper:
    def __init__(self, index_graph):
        self.index_graph = index_graph


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
