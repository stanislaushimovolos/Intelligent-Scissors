import numpy as np
from typing import Sequence
from collections import deque
from scipy.ndimage.filters import gaussian_filter1d
from skimage.filters import gaussian, laplace, sobel_h, sobel_v, sobel

from scissors.utils import unfold, create_spatial_feats, flatten_first_dims, norm_by_max_value, quadratic_kernel

default_params = {
    'laplace': 0.3,
    'direction': 0.2,
    'magnitude': 0.2,
    'local': 0.1,
    'inner': 0.1,
    'outer': 0.1,
    'maximum_cost': 255,
    'laplace_kernels': [3, 5, 7],
    'gaussian_kernels': [5, 5, 5],
    'laplace_weights': [0.2, 0.3, 0.4]
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

    def __call__(self, image: np.array):
        return self.get_static_costs(image)

    def get_static_costs(self, image: np.array):
        l_cost = self.get_laplace_cost(image, self.laplace_kernels, self.gaussian_kernels, self.laplace_weights)
        l_cost = unfold(l_cost, 3)
        l_cost = np.ceil(self.laplace_w * l_cost)

        d_cost = self.get_direction_cost(image)
        d_cost = np.ceil(self.direction_w * d_cost)
        total_cost = np.squeeze(l_cost + d_cost)
        return total_cost

    def get_laplace_cost(self, image, laplace_kernels: Sequence, gaussian_kernels: Sequence, weights: Sequence):
        total_cost = np.zeros((1,) + image.shape)
        for laplace_kernel, gaussian_kernel, w in zip(laplace_kernels, gaussian_kernels, weights):
            total_cost += w * self.calculate_single_laplace_cost(image, laplace_kernel, gaussian_kernel)
        return total_cost

    @staticmethod
    def calculate_single_laplace_cost(image: np.array, laplace_kernel_sz: int, gaussian_kernel: int):
        blurred = gaussian(image, gaussian_kernel)
        laplace_map = laplace(blurred, ksize=laplace_kernel_sz)
        laplace_map = np.expand_dims(laplace_map, 0)
        # create map of neighbouring pixels costs
        cost_map = unfold(laplace_map, np.array([laplace_kernel_sz, laplace_kernel_sz]))
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

        unfolded_grads = unfold(grads, 3)
        grads = grads[:, None, None, ...]

        spatial_feats = create_spatial_feats(image.shape, 3)
        link_feats = np.einsum('i..., i...', spatial_feats, grads)
        local_feats = np.abs(link_feats)

        sign_mask = np.sign(link_feats)
        distant_feats = sign_mask * np.einsum('i..., i...', spatial_feats, unfolded_grads)
        total_cost = 2 / (3 * np.pi) * (np.arccos(local_feats) + np.arccos(distant_feats))
        return total_cost


def get_dynamic_features(image, n_values=255, eps=1e-6):
    local_feats = norm_by_max_value(image, n_values)

    grads = -np.array([sobel_h(image), sobel_v(image)])
    grads /= (np.linalg.norm(grads) + eps)
    grads = grads[:, None, None, ...]

    spatial_feats = create_spatial_feats(image.shape, 3)
    dots_products = np.einsum('i..., i...', grads, spatial_feats)
    dots_products = flatten_first_dims(dots_products)

    unfolded_feats = unfold(np.expand_dims(local_feats, 0), 3)
    unfolded_feats = flatten_first_dims(np.squeeze(unfolded_feats, 0))

    inner_feats = get_inner_feats(unfolded_feats, dots_products)
    outer_feats = get_outer_feats(unfolded_feats, dots_products)

    inner_feats = np.ceil(inner_feats).astype(np.int)
    outer_feats = np.ceil(outer_feats).astype(np.int)
    local_feats = np.ceil(local_feats).astype(np.int)
    return local_feats, inner_feats, outer_feats


def get_outer_feats(feats, dots_products):
    indices = np.argmin(dots_products, axis=0).astype(np.int)
    return np.choose(indices, feats)


def get_inner_feats(feats, dots_products):
    indices = np.argmax(dots_products, axis=0).astype(np.int)
    return np.choose(indices, feats)


def get_magnitude_features(image: np.array, n_values: int):
    grads = sobel(image)
    grads = grads - np.min(grads)
    cost = 1 - grads / np.max(grads)
    cost = np.ceil((n_values - 1) * cost).astype(np.int)
    return cost


class CostProcessor:
    def __init__(self, image: np.array, hist_std=3, image_std=1, n_image_values=255, n_magnitude_values=196,
                 magnitude_w=None, inner_w=None, outer_w=None, local_w=None, maximum_cost=None):
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

        inner_w = inner_w or default_params['inner']
        outer_w = outer_w or default_params['outer']
        local_w = local_w or default_params['local']
        magnitude_w = magnitude_w or default_params['magnitude']
        maximum_cost = maximum_cost or default_params['maximum_cost']

        self.inner_weight = inner_w * maximum_cost
        self.outer_weight = outer_w * maximum_cost
        self.local_weight = local_w * maximum_cost
        self.magnitude_weight = magnitude_w * maximum_cost

        self.hist_std = hist_std
        self.image_std = image_std
        self.n_image_values = n_image_values
        self.n_magnitude_values = n_magnitude_values

        self.magnitude_feats = get_magnitude_features(gaussian(image, self.image_std), self.n_magnitude_values)
        self.local_feats, self.inner_feats, self.outer_feats \
            = get_dynamic_features(image, self.n_image_values)

    def compute(self, series):
        # local_hist = self.get_hist(series, self.local_feats, self.local_weight, self.n_image_values)
        # inner_hist = self.get_hist(series, self.inner_feats, self.inner_weight, self.n_image_values)
        # outer_hist = self.get_hist(series, self.outer_feats, self.outer_weight, self.n_image_values)
        magnitude_hist = self.get_hist(
            series, self.magnitude_feats, self.magnitude_weight,
            self.n_magnitude_values, self.hist_std
        )
        # local_cost = local_hist[self.local_feats]
        # inner_cost = inner_hist[self.inner_feats]
        # outer_cost = outer_hist[self.outer_feats]
        # local_cost + inner_cost + outer_cost

        magnitude_cost = magnitude_hist[self.magnitude_feats]
        neighbour_weights = np.array([
            [1, 0.707, 1],
            [0.707, 1, 0.707],
            [1, 0.707, 1]])

        neighbour_weights = neighbour_weights[None, :, :, None, None]
        magnitude_cost = unfold(magnitude_cost[None], 3)
        magnitude_cost = (neighbour_weights * magnitude_cost).squeeze(0)

        total_cost = (magnitude_cost + 0).astype(np.int)
        return total_cost

    @staticmethod
    def get_hist(series, feats, weight, n_values, std):
        hist = np.zeros(n_values)
        for i, idx in enumerate(series):
            hist[feats[idx]] += quadratic_kernel(i, len(series))

        hist = gaussian_filter1d(hist, std)
        hist = np.ceil(weight * (1 - hist / np.max(hist)))
        return hist


class Scissors:
    def __init__(self, static_cost, cost_processor, finder, capacity=32):
        """
        Parameters
        ----------
        static_cost : np.array
            array of shape (3, 3, height, width)
        cost_processor : CostProcessor
        finder : PathFinder
        capacity : int
            number of last pixels used for dynamic cost calculation
        """
        self.capacity = capacity
        self.path_finder = finder
        self.static_cost = static_cost

        self.current_dynamic_cost = None
        self.processor = cost_processor
        self.processed_pixels = deque(maxlen=self.capacity)

    def find_path(self, seed_x, seed_y, free_x, free_y):
        if len(self.processed_pixels) != 0:
            self.current_dynamic_cost = self.processor.compute(self.processed_pixels)

        path = self.path_finder.find_path(seed_x, seed_y, free_x, free_y, self.current_dynamic_cost)
        self.processed_pixels.extend(path)
        return path
