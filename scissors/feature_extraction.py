import numpy as np

from typing import Sequence
from scipy.ndimage.filters import gaussian_filter1d
from skimage.filters import gaussian, laplace, sobel_h, sobel_v, sobel

from scissors.utils import unfold, create_spatial_feats, flatten_first_dims, quadratic_kernel

default_params = {
    # static params
    'laplace': 0.4,
    'direction': 0.2,
    'magnitude': 0.3,
    'local': 0.1,
    'laplace_kernels': [3, 5, 7],
    'gaussian_kernels': [5, 5, 5],
    'laplace_weights': [0.2, 0.3, 0.5],

    # dynamic params
    'hist_std': 2,
    'image_std': 1,
    'history_capacity': 16,
    'n_image_values': 255,
    'n_magnitude_values': 255,

    # other params
    'maximum_cost': 512,
    'snap_scale': 3
}


class StaticExtractor:
    def __init__(self, laplace_kernels=None, laplace_weights=None, gaussian_kernels=None,
                 laplace_w=None, direction_w=None, maximum_cost=None):
        """
        Class for computing static features.

        Parameters
        ----------
        laplace_kernels : Sequence[int]
            defines the size of the laplace kernels.
        gaussian_kernels : Sequence[int]
            standard deviation for gaussian kernel.
        laplace_weights : Sequence[float]
            defines strength of different laplace filters
        laplace_w : float
            weight of laplace zero-crossing  features
        direction_w : float
            weight of gradient direction features
        maximum_cost : float
           specifies the largest possible integer cost
        """
        laplace_w = laplace_w or default_params['laplace']
        direction_w = direction_w or default_params['direction']
        maximum_cost = maximum_cost or default_params['maximum_cost']

        self.maximum_cost = maximum_cost
        self.laplace_w = laplace_w * maximum_cost
        self.direction_w = direction_w * maximum_cost

        self.laplace_weights = laplace_weights or default_params['laplace_weights']
        self.laplace_kernels = laplace_kernels or default_params['laplace_kernels']
        self.gaussian_kernels = gaussian_kernels or default_params['gaussian_kernels']

        assert len(self.laplace_weights) == len(self.laplace_kernels) == len(self.gaussian_kernels), \
            "Sequences must have equal length."

    def __call__(self, image: np.array, brightness: np.array):
        l_cost = self.get_laplace_cost(image, self.laplace_kernels, self.gaussian_kernels, self.laplace_weights)
        l_cost = unfold(l_cost)
        l_cost = np.ceil(self.laplace_w * l_cost)

        d_cost = self.get_direction_cost(brightness)
        d_cost = np.ceil(self.direction_w * d_cost)
        total_cost = np.squeeze(l_cost + d_cost)
        return total_cost

    def get_laplace_cost(self, image, laplace_kernels: Sequence, gaussian_kernels: Sequence, weights: Sequence):
        n_channels, *shape = image.shape
        total_cost = np.zeros((n_channels,) + tuple(shape))

        for i, channel in enumerate(image):
            for laplace_kernel, gaussian_kernel, w in zip(laplace_kernels, gaussian_kernels, weights):
                total_cost[i] = w * self.calculate_single_laplace_cost(channel, laplace_kernel, gaussian_kernel)

        total_cost = np.max(total_cost, axis=0, keepdims=True)
        return total_cost

    @staticmethod
    def calculate_single_laplace_cost(image: np.array, laplace_kernel_sz: int, gaussian_kernel: int):
        blurred = gaussian(image, gaussian_kernel)
        laplace_map = laplace(blurred, ksize=laplace_kernel_sz)
        laplace_map = laplace_map[None]
        # create map of neighbouring pixels costs
        cost_map = unfold(laplace_map)
        cost_map = flatten_first_dims(np.squeeze(cost_map))
        # leave only direct neighbours
        cost_map = cost_map[[1, 3, 5, 7], :, :]
        # get max elements with the opposite sign
        signs = np.sign(laplace_map)
        opposites = cost_map * (cost_map * signs < 0)
        opposites = np.max(np.abs(opposites), axis=0)
        output = np.abs(laplace_map) > opposites
        return output

    @staticmethod
    def get_direction_cost(image: np.array, eps=1e-6):
        grads = np.stack([sobel_v(image), -sobel_h(image)])
        grads /= (np.linalg.norm(grads, axis=0) + eps)

        unfolded_grads = unfold(grads)
        grads = grads[:, None, None, ...]

        spatial_feats = create_spatial_feats(image.shape)
        link_feats = np.einsum('i..., i...', spatial_feats, grads)
        local_feats = np.abs(link_feats)

        sign_mask = np.sign(link_feats)
        distant_feats = sign_mask * np.einsum('i..., i...', spatial_feats, unfolded_grads)
        total_cost = 2 / (3 * np.pi) * (np.arccos(local_feats) + np.arccos(distant_feats))
        return total_cost


class CostProcessor:
    def __init__(self, image: np.array, brightness: np.array, hist_std=None, image_std=None, n_image_values=None,
                 n_magnitude_values=None, magnitude_w=None, inner_w=None, outer_w=None, local_w=None,
                 maximum_cost=None):
        """
        Class for computing dynamic features histograms.

        Parameters
        ----------
        image: np.array
            input image
        hist_std : float
            standard deviation of gaussian kernel used for smoothing dynamic histograms
        n_image_values : int
            defines a possible range of values of dynamic features
        magnitude_w : float
            weight of gradient magnitude features
        inner_w : float
            weight of inner features
        outer_w : float
            weight of outer features
        local_w : float
            weight of local features
        maximum_cost : int
            specifies the largest possible integer cost
        """

        # inner_w = inner_w or default_params['inner']
        # outer_w = outer_w or default_params['outer']
        hist_std = hist_std or default_params['hist_std']
        image_std = image_std or default_params['image_std']
        n_image_values = n_image_values or default_params['n_image_values']
        n_magnitude_values = n_magnitude_values or default_params['n_magnitude_values']

        local_w = local_w or default_params['local']
        magnitude_w = magnitude_w or default_params['magnitude']
        maximum_cost = maximum_cost or default_params['maximum_cost']

        # self.inner_weight = inner_w * maximum_cost
        # self.outer_weight = outer_w * maximum_cost
        self.local_weight = local_w * maximum_cost
        self.magnitude_weight = magnitude_w * maximum_cost

        self.hist_std = hist_std
        self.image_std = image_std
        self.n_image_values = n_image_values
        self.n_magnitude_values = n_magnitude_values

        self.brightness = brightness
        self.magnitude_feats = self.get_magnitude_features(image, self.n_magnitude_values, self.image_std)
        self.local_feats = self.get_local_features(gaussian(brightness, self.image_std), self.n_image_values)

    def compute(self, series):
        local_hist = self.get_hist(
            series, self.local_feats, self.local_weight,
            self.n_image_values, self.hist_std
        )
        magnitude_hist = self.get_hist(
            series, self.magnitude_feats, self.magnitude_weight,
            self.n_magnitude_values, self.hist_std
        )

        local_cost = local_hist[self.local_feats]
        magnitude_cost = magnitude_hist[self.magnitude_feats]
        neighbour_weights = np.array([
            [1, 0.707, 1],
            [0.707, 1, 0.707],
            [1, 0.707, 1]])

        neighbour_weights = neighbour_weights[None, :, :, None, None]
        magnitude_cost = (neighbour_weights * magnitude_cost)
        total_cost = (magnitude_cost + local_cost).squeeze(0).astype(np.int)
        return total_cost

    @staticmethod
    def get_hist(series, feats, weight, n_values, std):
        hist = np.zeros(n_values)
        for i, idx in enumerate(series):
            y, x = idx
            hist[feats[:, 1, 1, y, x]] += quadratic_kernel(i, len(series))

        hist = gaussian_filter1d(hist, std)
        hist = np.ceil(weight * (1 - hist / np.max(hist)))
        return hist

    # TODO: implement other features
    @staticmethod
    def get_local_features(image, n_values, eps=1e-6):
        local_feats = image / np.max(image)

        # grads = -np.array([sobel_h(image), sobel_v(image)])
        # grads /= (np.linalg.norm(grads) + eps)
        # grads = grads[:, None, None, ...]

        # spatial_feats = create_spatial_feats(image.shape)
        # dot_products = np.einsum('i..., i...', grads, spatial_feats)
        # dot_products = flatten_first_dims(dot_products)
        #
        # unfolded_feats = unfold(local_feats[None])
        # unfolded_feats = flatten_first_dims(np.squeeze(unfolded_feats, 0))

        # def get_outer_feats(feats, dots_products):
        #     indices = np.argmin(dots_products, axis=0).astype(np.int)
        #     return np.choose(indices, feats)
        #
        #
        # def get_inner_feats(feats, dots_products):
        #     indices = np.argmax(dots_products, axis=0).astype(np.int)
        #     return np.choose(indices, feats)

        # inner_feats = get_inner_feats(unfolded_feats, dot_products)
        # outer_feats = get_outer_feats(unfolded_feats, dot_products)

        # inner_feats = np.ceil((n_values - 1) * inner_feats).astype(np.int)
        # outer_feats = np.ceil((n_values - 1) * outer_feats).astype(np.int)
        local_feats = np.ceil((n_values - 1) * local_feats).astype(np.int)
        local_feats = unfold(local_feats[None]).astype(np.int)
        return local_feats

    @staticmethod
    def get_magnitude_features(image: np.array, n_values: int, std: float):
        n_channels, *shape = image.shape
        grads = np.zeros((n_channels,) + tuple(shape))

        # process each RGB channel
        for i, channel in enumerate(image):
            channel = gaussian(channel, std)
            grads[i] = sobel(channel)

        # choose maximum over the channels
        grads = np.max(grads, axis=0, keepdims=True)
        grads = grads - np.min(grads)

        cost = 1 - grads / np.max(grads)
        cost = np.ceil((n_values - 1) * cost).astype(np.int)
        cost = unfold(cost).astype(int)
        return cost


class Scissors:
    def __init__(self, static_cost, dynamic_processor, finder, capacity=None):
        """
        Parameters
        ----------
        static_cost : np.array
            array of shape (3, 3, height, width)
        dynamic_processor : CostProcessor
        finder : PathFinder
        capacity : int
            number of last pixels used for dynamic cost calculation
        """
        self.capacity = capacity or default_params['history_capacity']
        self.path_finder = finder
        self.static_cost = static_cost

        self.current_dynamic_cost = None
        self.processor = dynamic_processor
        self.grads_map = sobel(dynamic_processor.brightness)
        self.processed_pixels = list()

    def find_path(self, seed_x, seed_y, free_x, free_y):
        if len(self.processed_pixels) != 0:
            self.current_dynamic_cost = self.processor.compute(self.processed_pixels)

        free_x, free_y = self.get_cursor_snap_point(free_x, free_y, self.grads_map)
        path = self.path_finder.find_path(seed_x, seed_y, free_x, free_y, self.current_dynamic_cost)
        self.processed_pixels.extend(reversed(path))

        if len(self.processed_pixels) > self.capacity:
            self.processed_pixels = self.processed_pixels[-self.capacity:]
        return path

    @staticmethod
    def get_cursor_snap_point(x, y, grads: np.array, snap_scale: int = 3):
        region = grads[y - snap_scale:y + snap_scale, x - snap_scale:x + snap_scale]

        max_grad_idx = np.unravel_index(region.argmax(), region.shape)
        max_grad_idx = np.array(max_grad_idx)

        y, x = max_grad_idx + np.array([y - snap_scale, x - snap_scale])
        return x, y
