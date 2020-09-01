import numpy as np

from typing import Sequence, Union
from scipy.ndimage.filters import gaussian_filter1d
from skimage.filters import gaussian, laplace, sobel_h, sobel_v, sobel

from scissors.search import search
from scissors.utils import unfold, create_spatial_feats, flatten_first_dims, quadratic_kernel, preprocess_image

# Parameters  depend ot the image size.
# These parameters were selected for 512 x 512 images.

default_params = {
    # static params
    # sum of all weights must be equal to 1
    'laplace': 0.3,
    'direction': 0.2,
    'magnitude': 0.2,
    'local': 0.1,
    'inner': 0.1,
    'outer': 0.1,
    'laplace_kernels': [3, 5, 7],
    'gaussian_kernel': 5,
    'laplace_weights': [0.2, 0.3, 0.5],

    # dynamic params
    'hist_std': 2,
    'image_std': 1,
    'distance_value': 3,
    'history_capacity': 16,
    'n_image_values': 255,
    'n_magnitude_values': 255,

    # other params
    'maximum_cost': 255,
    'snap_scale': 3
}


class StaticExtractor:
    def __init__(self, laplace_kernels=None, laplace_weights=None, std=None,
                 laplace_w=None, direction_w=None, maximum_cost=None):
        """
        Class for computing static features.

        Parameters
        ----------
        laplace_kernels : Sequence[int]
            defines the size of the laplace kernels.
        std : int
            standard deviation for gaussian kernel.
        laplace_weights : Sequence[float]
            defines strength of different laplace filters
        laplace_w : float
            weight of laplace zero-crossing  features
        direction_w : float
            weight of gradient direction features
        maximum_cost : int
           specifies the largest possible integer cost
        """

        std = std or default_params['gaussian_kernel']
        laplace_w = laplace_w or default_params['laplace']
        direction_w = direction_w or default_params['direction']
        maximum_cost = maximum_cost or default_params['maximum_cost']

        self.std = std
        self.maximum_cost = maximum_cost
        self.laplace_w = laplace_w * maximum_cost
        self.direction_w = direction_w * maximum_cost

        self.laplace_weights = laplace_weights or default_params['laplace_weights']
        self.laplace_kernels = laplace_kernels or default_params['laplace_kernels']

        assert len(self.laplace_weights) == len(self.laplace_kernels), \
            "Sequences must have equal length."

    def __call__(self, image: np.array, brightness: np.array) -> np.array:
        # calculate laplace cost
        l_cost = self.get_laplace_cost(image)
        l_cost = unfold(l_cost)
        l_cost = np.ceil(self.laplace_w * l_cost)
        # calculate direction costs
        d_cost = self.get_direction_cost(brightness)
        d_cost = np.ceil(self.direction_w * d_cost)
        # calculate total static cost
        total_cost = np.squeeze(l_cost + d_cost)
        return total_cost

    def get_laplace_cost(self, image: np.array) -> np.array:
        n_channels, *shape = image.shape
        total_cost = np.zeros((n_channels,) + tuple(shape))

        # smooth image
        image = gaussian(image, self.std)
        # calculate zero crossings for each kernel and channel
        for i, channel in enumerate(image):
            for laplace_kernel, w in zip(self.laplace_kernels, self.laplace_weights):
                total_cost[i] = w * self.calculate_single_laplace_cost(channel, laplace_kernel)

        # maximize over the channels
        total_cost = np.max(total_cost, axis=0, keepdims=True)
        return total_cost

    @staticmethod
    def calculate_single_laplace_cost(image: np.array, laplace_kernel_sz: int) -> np.array:
        laplace_map = laplace(image, ksize=laplace_kernel_sz)
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
    def get_direction_cost(image: np.array, eps=1e-6) -> np.array:
        # calculate vectors perpendicular to gradients
        grads = np.stack([sobel_v(image), -sobel_h(image)])
        grads /= (np.linalg.norm(grads, axis=0) + eps)

        unfolded_grads = unfold(grads)
        grads = grads[:, None, None, ...]

        # calculate dot products
        spatial_feats = create_spatial_feats(image.shape)
        link_feats = np.einsum('i..., i...', spatial_feats, grads)
        # get d_p features
        local_feats = np.abs(link_feats)
        # get d_q features
        sign_mask = np.sign(link_feats)
        distant_feats = sign_mask * np.einsum('i..., i...', spatial_feats, unfolded_grads)
        # calculate total gradient direction cost
        total_cost = 2 / (3 * np.pi) * (np.arccos(local_feats) + np.arccos(distant_feats))
        return total_cost


class CostProcessor:
    def __init__(self, image: np.array, brightness: np.array, hist_std=None, image_std=None, distance_value=None,
                 n_image_values=None, n_magnitude_values=None, magnitude_w=None, inner_w=None, outer_w=None,
                 local_w=None, maximum_cost=None):
        """
        Class for computing dynamic features.

        Parameters
        ----------
        image: np.array
            input image
        hist_std : float
            size of gaussian kernel used for smoothing dynamic histograms
        image_std: float
            size of gaussian kernel used for smoothing the image
        distance_value : int
            distance for inner/outer pixels
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
        hist_std = hist_std or default_params['hist_std']
        image_std = image_std or default_params['image_std']
        distance_value = distance_value or default_params['distance_value']

        n_image_values = n_image_values or default_params['n_image_values']
        n_magnitude_values = n_magnitude_values or default_params['n_magnitude_values']

        local_w = local_w or default_params['local']
        magnitude_w = magnitude_w or default_params['magnitude']
        maximum_cost = maximum_cost or default_params['maximum_cost']

        self.inner_weight = inner_w * maximum_cost
        self.outer_weight = outer_w * maximum_cost
        self.local_weight = local_w * maximum_cost
        self.magnitude_weight = magnitude_w * maximum_cost

        self.hist_std = hist_std
        self.n_image_values = n_image_values
        self.n_magnitude_values = n_magnitude_values

        self.brightness = brightness
        self.magnitude_feats = self.get_magnitude_features(image, n_magnitude_values, image_std)

        smoothed = gaussian(brightness, image_std)
        self.local_feats = self.get_direct_pixel_feats(smoothed, n_image_values)
        self.inner_feats, self.outer_feats = self.get_externals_pixel_feats(smoothed, n_image_values, distance_value)

    def __call__(self, series: Sequence[tuple]) -> np.array:
        return self.compute_dynamic_cost(series)

    def compute_dynamic_cost(self, series: Sequence[tuple]) -> np.array:
        # calculate histograms
        local_hist = self.get_hist(
            series, self.local_feats, self.local_weight,
            self.n_image_values, self.hist_std
        )
        inner_hist = self.get_hist(
            series, self.inner_feats, self.inner_weight,
            self.n_image_values, self.hist_std
        )
        outer_hist = self.get_hist(
            series, self.outer_feats, self.outer_weight,
            self.n_image_values, self.hist_std
        )
        magnitude_hist = self.get_hist(
            series, self.magnitude_feats, self.magnitude_weight,
            self.n_magnitude_values, self.hist_std
        )
        # calculate dynamic costs
        local_cost = local_hist[self.local_feats]
        inner_cost = inner_hist[self.inner_feats]
        outer_cost = outer_hist[self.outer_feats]

        magnitude_cost = magnitude_hist[self.magnitude_feats]
        neighbour_weights = np.array([
            [1, 0.707, 1],
            [0.707, 1, 0.707],
            [1, 0.707, 1]])

        neighbour_weights = neighbour_weights[None, :, :, None, None]
        magnitude_cost = (neighbour_weights * magnitude_cost)

        total_cost = (magnitude_cost + local_cost + inner_cost + outer_cost)
        total_cost = total_cost.squeeze(0).astype(np.int)
        return total_cost

    @staticmethod
    def get_hist(series: Sequence[tuple], feats: np.array, weight: float, n_values: int, std: int):
        hist = np.zeros(n_values)
        for i, idx in enumerate(series):
            y, x = idx
            hist[feats[:, 1, 1, y, x]] += quadratic_kernel(i, len(series))

        hist = gaussian_filter1d(hist, std)
        hist = np.ceil(weight * (1 - hist / np.max(hist)))
        return hist

    @staticmethod
    def get_direct_pixel_feats(image: np.array, n_values: int) -> np.array:
        local_feats = image / np.max(image)
        local_feats = np.ceil((n_values - 1) * local_feats).astype(np.int)
        local_feats = unfold(local_feats[None]).astype(np.int)
        return local_feats

    @staticmethod
    def get_externals_pixel_feats(image: np.array, n_values: int, k_distance: int, eps=1e-4) -> np.array:
        h, w = image.shape
        grads_map = np.array([sobel_h(image), sobel_v(image)])
        grads_map = grads_map / (np.linalg.norm(grads_map, axis=0) + eps)

        def get_shifted_feats(directions):
            shifts = np.round(directions * k_distance).astype(np.int)
            grid = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'))

            # calculate coords of inner/outer pixels
            coords = grid + shifts
            # clip values
            coords[coords < 0] = 0
            coords[:, :, -k_distance:] = np.clip(coords[:, :, -k_distance:], 0, w - 1)
            coords[:, -k_distance:, :] = np.clip(coords[:, -k_distance:, :], 0, h - 1)
            # get required pixels
            feats = image[coords[0].reshape(-1), coords[1].reshape(-1)].reshape(h, w)
            feats = feats / (np.max(feats) + eps)
            feats = np.ceil((n_values - 1) * feats)
            feats = unfold(feats[None]).astype(np.int)
            return feats

        outer_feats = get_shifted_feats(grads_map)
        inner_feats = get_shifted_feats(-grads_map)
        return inner_feats, outer_feats

    @staticmethod
    def get_magnitude_features(image: np.array, n_values: int, std: int) -> np.array:
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
    def __init__(self, image: np.array, capacity=None, use_dynamic_features=True):
        """
        Parameters
        ----------
        image : np.array
            array of shape (3, 3, height, width)
        capacity : int
            number of last pixels used for dynamic cost calculation
        """

        image, brightness = preprocess_image(image)
        static_extractor = StaticExtractor()
        static_cost = static_extractor(image, brightness)

        self.static_cost = static_cost.astype(np.int)
        self.maximum_cost = static_extractor.maximum_cost
        self.capacity = capacity or default_params['history_capacity']

        self.current_dynamic_cost = None
        self.processor = CostProcessor(image, brightness) if use_dynamic_features else lambda x: None

        self.grads_map = sobel(brightness)
        self.processed_pixels = list()

    def find_path(self, seed_x: int, seed_y: int, free_x: int, free_y: int) -> Sequence[tuple]:
        if len(self.processed_pixels) != 0:
            self.current_dynamic_cost = self.processor(self.processed_pixels)

        free_x, free_y = self.get_cursor_snap_point(free_x, free_y, self.grads_map)
        path = self.calculate_segment(
            seed_x, seed_y, free_x, free_y,
            self.maximum_cost, self.current_dynamic_cost, self.static_cost
        )
        self.processed_pixels.extend(reversed(path))
        if len(self.processed_pixels) > self.capacity:
            self.processed_pixels = self.processed_pixels[-self.capacity:]
        return path

    @staticmethod
    def calculate_segment(seed_x: int, seed_y: int, free_x: int, free_y: int, maximum_cost: int,
                          dynamic_cost: Union[np.array, None], static_cost: np.array) -> Sequence[tuple]:
        h, w = static_cost.shape[2:]
        if dynamic_cost is None:
            dynamic_cost = np.zeros((3, 3, h, w), dtype=np.int)

        node_map = search(static_cost, dynamic_cost, w, h, seed_x, seed_y, maximum_cost)
        cur_x, cur_y = node_map[:, free_x, free_y]

        history = []
        while (cur_x, cur_y) != (seed_x, seed_y):
            history.append((cur_y, cur_x))
            cur_x, cur_y = node_map[:, cur_x, cur_y]
        return history

    @staticmethod
    def get_cursor_snap_point(x: int, y: int, grads: np.array, snap_scale: int = 3):
        region = grads[y - snap_scale:y + snap_scale, x - snap_scale:x + snap_scale]

        max_grad_idx = np.unravel_index(region.argmax(), region.shape)
        max_grad_idx = np.array(max_grad_idx)
        y, x = max_grad_idx + np.array([y - snap_scale, x - snap_scale])
        return x, y
