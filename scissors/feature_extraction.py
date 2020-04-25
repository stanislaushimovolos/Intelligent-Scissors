import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from skimage.filters import gaussian, laplace, sobel_h, sobel_v

from scissors.utils import unfold, create_spatial_feats, flatten_first_dims, norm_by_max_value

default_params = {
    'laplace': 0.3,
    'direction': 0.1,
    'magnitude': 0.3,
    'local': 0.1,
    'inner': 0.1,
    'outer': 0.1,
    'maximum_cost': 4096,
}


class StaticExtractor:
    def __init__(self, std=2, laplace_filter_size=5, params=None, filter_size=3):
        if params is None:
            params = default_params

        # TODO replace by default params
        self.laplace_w = params['laplace']
        self.magnitude_w = params['magnitude']
        self.direction_w = params['direction']
        self.maximum_cost = params['maximum_cost']

        self.gaussian_std = std
        self.laplace_filter_size = laplace_filter_size
        self.filter_size = np.array([filter_size, filter_size])

    def __call__(self, image):
        return self.get_total_link_costs(image)

    def get_total_link_costs(self, image):
        l_cost = self.get_laplace_cost(image, self.laplace_filter_size, self.gaussian_std)
        l_cost = unfold(l_cost, self.filter_size)
        l_cost = np.ceil(self.laplace_w * self.maximum_cost * l_cost)

        g_cost = self.get_magnitude_cost(image)
        g_cost = unfold(g_cost, self.filter_size)
        g_cost = np.ceil(self.magnitude_w * self.maximum_cost * g_cost)

        d_cost = self.get_direction_cost(image)
        d_cost = np.ceil(self.direction_w * self.maximum_cost * d_cost)
        total_cost = l_cost + g_cost + d_cost
        return np.squeeze(total_cost)

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

    def get_direction_cost(self, img, eps=1e-5):
        # TODO new check
        grads = np.stack([-sobel_h(img), sobel_v(img)])
        grads /= np.linalg.norm(grads, axis=0) + eps

        unfolded_grads = unfold(grads, self.filter_size)
        grads = grads[:, None, None, ...]

        spatial_feats = create_spatial_feats(img.shape, self.filter_size)
        link_feats = np.einsum('i..., i...', spatial_feats, grads)
        local_feats = np.abs(link_feats)

        sign_mask = np.sign(link_feats)
        distant_feats = sign_mask * np.einsum('i..., i...', spatial_feats, unfolded_grads)
        total_cost = 2 / (3 * np.pi) * (np.arccos(local_feats) + np.arccos(distant_feats))
        return total_cost


class DynamicExtractor:
    def __init__(self, filter_size=3, n_values=255):
        self.filter_size = np.array([filter_size, filter_size])
        self.n_values = n_values

    def extract_features(self, image):
        local_feats = norm_by_max_value(image, self.n_values)

        grads = np.array([sobel_h(image), sobel_v(image)])

        # TODO check
        grads /= np.linalg.norm(grads)
        grads = grads[:, None, None, ...]

        unfolded_feats = unfold(np.expand_dims(local_feats, 0), self.filter_size)
        unfolded_feats = flatten_first_dims(np.squeeze(unfolded_feats, 0))

        spatial_feats = create_spatial_feats(image.shape, self.filter_size)
        dots_products = np.einsum('i..., i...', grads, spatial_feats)
        dots_products = flatten_first_dims(dots_products)

        inner_feats = self.get_inner_feats(unfolded_feats, dots_products)
        outer_feats = self.get_outer_feats(unfolded_feats, dots_products)

        inner_feats = np.clip(np.ceil(inner_feats), 0, self.n_values - 1)
        outer_feats = np.clip(np.ceil(outer_feats), 0, self.n_values - 1)
        local_feats = np.clip(np.ceil(local_feats), 0, self.n_values - 1)
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
    def __init__(self, local_feats, inner_feats, outer_feats, params=None, filter_size=3, std=3, n_values=255):
        if params is None:
            params = default_params

        # TODO replace by default params
        self.maximum_cost = params['maximum_cost']
        self.inner_weight = self.maximum_cost * params['inner']
        self.outer_weight = self.maximum_cost * params['outer']
        self.local_weight = self.maximum_cost * params['local']

        self.std = std
        self.n_values = n_values
        self.filter_size = np.array([filter_size, filter_size])

        self.local_feats = local_feats.astype(np.int)
        self.inner_feats = inner_feats.astype(np.int)
        self.outer_feats = outer_feats.astype(np.int)

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

        # TODO weighted function
        for i, idx in enumerate(series):
            hist[feats[idx]] += 1

        hist = gaussian_filter1d(hist, self.std)
        hist = np.ceil(weight * (1 - hist / np.max(hist)))
        return hist


class Scissors:
    def __init__(self, static_cost, dynamic_feats, finder, use_dynamic_feats=True):
        self.use_dynamic_feats = use_dynamic_feats
        self.static_cost = static_cost
        self.path_finder = finder

        # TODO use DI
        self.processor = CostProcessor(*dynamic_feats)
        self.current_dynamic_cost = None
        self.processed_pixels = None

    def get_dynamic_cost(self, u, v, edge, prev_edge):
        dynamic_addition = 0
        if self.current_dynamic_cost is not None:
            dynamic_addition = self.current_dynamic_cost[self.path_finder.get_pos_from_node(v)]
        return edge + dynamic_addition

    def find_path(self, seed_point, free_point):
        if self.processed_pixels is not None:
            self.current_dynamic_cost = self.processor.compute(self.processed_pixels)

        path = self.path_finder.find_path(seed_point, free_point, self.get_dynamic_cost)
        path = [np.flip(cor) for cor in path]

        # TODO place by deque/circular buffer
        self.processed_pixels = path
        return path
