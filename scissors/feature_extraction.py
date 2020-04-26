import numpy as np
from collections import deque
from scipy.ndimage.filters import gaussian_filter1d
from skimage.filters import gaussian, laplace, sobel_h, sobel_v, sobel

from scissors.utils import unfold, create_spatial_feats, flatten_first_dims, norm_by_max_value, get_pos_from_node, \
    quadratic_kernel

default_params = {
    'laplace': 0.4,
    'direction': 0.3,
    'magnitude': 0.4,
    'local': 0.1,
    'inner': 0.1,
    'outer': 0.1,
    'maximum_cost': 8192,
}


class StaticExtractor:
    def __init__(self, laplace_kernel_sz=3, gaussian_std=4, filter_size=3, laplace_w=None, magnitude_w=None,
                 direction_w=None,
                 maximum_cost=None):
        if laplace_w is None:
            laplace_w = default_params['laplace']

        if magnitude_w is None:
            magnitude_w = default_params['magnitude']

        if direction_w is None:
            direction_w = default_params['direction']

        if maximum_cost is None:
            maximum_cost = default_params['maximum_cost']

        self.laplace_w = laplace_w * maximum_cost
        self.magnitude_w = magnitude_w * maximum_cost
        self.direction_w = direction_w * maximum_cost

        self.gaussian_std = gaussian_std
        self.laplace_kernel_sz = laplace_kernel_sz
        self.filter_size = np.array([filter_size, filter_size])

    def __call__(self, image):
        return self.get_total_link_costs(image)

    def get_total_link_costs(self, image):
        l_cost = self.get_laplace_cost(image, self.laplace_kernel_sz, self.gaussian_std)
        l_cost = unfold(l_cost, self.filter_size)
        l_cost = np.ceil(self.laplace_w * l_cost)

        g_cost = self.get_magnitude_cost(image)
        g_cost = unfold(g_cost, self.filter_size)
        g_cost = np.ceil(self.magnitude_w * g_cost)

        d_cost = self.get_direction_cost(image)
        d_cost = np.ceil(self.direction_w * d_cost)
        total_cost = np.squeeze(l_cost + g_cost + d_cost)
        return total_cost

    @staticmethod
    def get_laplace_cost(image, laplace_kernel_sz, gaussian_std):
        # TODO add several kernels
        output = np.ones_like(image)
        blurred = gaussian(image, gaussian_std)
        laplace_map = laplace(blurred, ksize=laplace_kernel_sz)

        unfolded_map = unfold(
            np.expand_dims(laplace_map, 0),
            np.array([laplace_kernel_sz, laplace_kernel_sz])
        )
        unfolded_map = flatten_first_dims(np.squeeze(unfolded_map))
        n_positive = np.sum(unfolded_map >= 0, axis=0)
        n_negative = np.sum(unfolded_map < 0, axis=0)

        zero_crossing_mask = ((n_negative > 0) & (n_positive > 0))
        output[zero_crossing_mask] = 0
        return np.expand_dims(output, 0)

    @staticmethod
    def get_magnitude_cost(image):
        grads = sobel(image)
        cost = 1 - grads / np.max(grads)
        cost = np.expand_dims(cost, 0)
        return cost

    def get_direction_cost(self, image, eps=1e-6):
        grads = np.stack([sobel_v(image), -sobel_h(image)])
        grads /= (np.linalg.norm(grads, axis=0) + eps)

        unfolded_grads = unfold(grads, self.filter_size)
        grads = grads[:, None, None, ...]

        spatial_feats = create_spatial_feats(image.shape, self.filter_size)
        link_feats = np.einsum('i..., i...', spatial_feats, grads)
        local_feats = np.abs(link_feats)

        sign_mask = np.sign(link_feats)
        distant_feats = sign_mask * np.einsum('i..., i...', spatial_feats, unfolded_grads)
        total_cost = 2 / (3 * np.pi) * (np.arccos(local_feats) + np.arccos(distant_feats))
        return total_cost


class DynamicExtractor:
    def __init__(self, filter_size=5, n_values=255):
        self.filter_size = np.array([filter_size, filter_size])
        self.n_values = n_values

    def __call__(self, image):
        return self.extract_features(image)

    def extract_features(self, image, eps=1e-6):
        local_feats = norm_by_max_value(image, self.n_values)

        grads = -np.array([sobel_h(image), sobel_v(image)])
        grads /= (np.linalg.norm(grads) + eps)
        grads = grads[:, None, None, ...]

        spatial_feats = create_spatial_feats(image.shape, self.filter_size)
        dots_products = np.einsum('i..., i...', grads, spatial_feats)
        dots_products = flatten_first_dims(dots_products)

        unfolded_feats = unfold(np.expand_dims(local_feats, 0), self.filter_size)
        unfolded_feats = flatten_first_dims(np.squeeze(unfolded_feats, 0))

        inner_feats = self.get_inner_feats(unfolded_feats, dots_products)
        outer_feats = self.get_outer_feats(unfolded_feats, dots_products)

        inner_feats = np.clip(np.ceil(inner_feats), 0, self.n_values - 1).astype(int)
        outer_feats = np.clip(np.ceil(outer_feats), 0, self.n_values - 1).astype(int)
        local_feats = np.clip(np.ceil(local_feats), 0, self.n_values - 1).astype(int)
        return local_feats, inner_feats, outer_feats

    @staticmethod
    def get_inner_feats(feats, dots_products):
        indices = np.argmax(dots_products, axis=0).astype(np.int)
        return np.choose(indices, feats)

    @staticmethod
    def get_outer_feats(feats, dots_products):
        indices = np.argmin(dots_products, axis=0).astype(np.int)
        return np.choose(indices, feats)


class CostProcessor:
    def __init__(self, local_feats, inner_feats, outer_feats, filter_size=3, std=3, n_values=255, maximum_cost=None,
                 inner_w=None, outer_w=None, local_w=None):
        if maximum_cost is None:
            maximum_cost = default_params['maximum_cost']

        if inner_w is None:
            inner_w = default_params['inner']

        if outer_w is None:
            outer_w = default_params['outer']

        if local_w is None:
            local_w = default_params['local']

        self.inner_weight = maximum_cost * inner_w
        self.outer_weight = maximum_cost * outer_w
        self.local_weight = maximum_cost * local_w

        self.std = std
        self.n_values = n_values
        self.filter_size = np.array([filter_size, filter_size])

        self.local_feats = local_feats
        self.inner_feats = inner_feats
        self.outer_feats = outer_feats

    def compute(self, series):
        local_hist = self.get_hist(series, self.local_feats, self.local_weight)
        inner_hist = self.get_hist(series, self.inner_feats, self.inner_weight)
        outer_hist = self.get_hist(series, self.outer_feats, self.outer_weight)

        local_cost = local_hist[self.local_feats]
        inner_cost = inner_hist[self.inner_feats]
        outer_cost = outer_hist[self.outer_feats]
        total_cost = local_cost + inner_cost + outer_cost
        return total_cost

    def get_hist(self, series, feats, weight):
        hist = np.zeros(self.n_values)

        for i, idx in enumerate(series):
            hist[feats[idx]] += quadratic_kernel(i, len(series))

        hist = gaussian_filter1d(hist, self.std)
        hist = np.ceil(weight * (1 - hist / np.max(hist)))
        return hist


class Scissors:
    def __init__(self, static_cost, dynamic_feats, finder, capacity=32):
        self.capacity = capacity
        self.path_finder = finder
        self.static_cost = static_cost

        self.current_dynamic_cost = None
        self.processor = CostProcessor(*dynamic_feats)
        self.processed_pixels = deque(maxlen=self.capacity)

    def get_dynamic_cost(self, u, v, edge, prev_edge):
        dynamic_addition = 0
        if self.current_dynamic_cost is not None:
            index = get_pos_from_node(v, self.path_finder.index_key)
            dynamic_addition = self.current_dynamic_cost[index]
        return edge + dynamic_addition

    def find_path(self, seed_point, free_point):
        if len(self.processed_pixels) != 0:
            self.current_dynamic_cost = self.processor.compute(self.processed_pixels)

        path = self.path_finder.find_path(seed_point, free_point, self.get_dynamic_cost)
        self.processed_pixels.extend(path)
        return path
