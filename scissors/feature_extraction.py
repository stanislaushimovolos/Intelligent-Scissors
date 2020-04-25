import numpy as np
from skimage.filters import gaussian, laplace, sobel_h, sobel_v
from scipy.ndimage.filters import gaussian_filter1d
from scissors.utils import unfold, create_spatial_feats, flatten_first_dims, norm_by_max

default_params = {
    'laplace': 0.3,
    'direction': 0.1,
    'magnitude': 0.3,
    'local': 0.1,
    'inner': 0.1,
    'outer': 0.1,
    'maximum_cost': 2048,
}


class StaticFeatureExtractor:
    def __init__(self, gaussian_std=2, laplace_filter_size=5, params=None, connected_area=3):
        if params is None:
            params = default_params

        self.laplace_w = params['laplace']
        self.magnitude_w = params['magnitude']
        self.direction_w = params['direction']
        self.maximum_cost = params['maximum_cost']

        self.gaussian_std = gaussian_std
        self.laplace_filter_size = laplace_filter_size
        self.filter_size = np.array([connected_area, connected_area])

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


class DynamicFeatureExtractor:
    def __init__(self, image, connected_area=3, max_feats_value=255):
        self.image_feats = norm_by_max(image, max_feats_value)
        self.filter_size = np.array([connected_area, connected_area])

        grads = np.array([sobel_h(image), sobel_v(image)])
        grads /= np.linalg.norm(grads)
        grads = grads[:, None, None, ...]

        unfolded_feats = unfold(np.expand_dims(self.image_feats, 0), self.filter_size)
        unfolded_feats = flatten_first_dims(np.squeeze(unfolded_feats, 0))

        spatial_feats = create_spatial_feats(image.shape, self.filter_size)
        dots_products = np.einsum('i..., i...', grads, spatial_feats)
        dots_products = flatten_first_dims(dots_products)

        # TODO add weights
        self.inner_feats = self.get_inner_feats(unfolded_feats, dots_products)
        self.outer_feats = self.get_outer_feats(unfolded_feats, dots_products)

        self.inner_feats = np.ceil(self.inner_feats)
        self.outer_feats = np.ceil(self.outer_feats)
        self.image_feats = np.ceil(self.image_feats)

    @staticmethod
    def create_histogram(series, max_value):
        freq, bins = np.histogram(series, np.arange(max_value))
        return freq

    @staticmethod
    def get_inner_feats(feats, dots_products):
        indices = np.argmax(dots_products, axis=0).astype(np.int)
        return np.choose(indices, feats)

    @staticmethod
    def get_outer_feats(feats, dots_products):
        indices = np.argmin(dots_products, axis=0).astype(np.int)
        return np.choose(indices, feats)


class DynamicFeaturesProcessor:
    def __init__(self, local_feats, inner_feats, outer_feats, params=None, train_segment_size=64, std=3, n_bins=255):
        if params is None:
            params = default_params

        self.maximum_cost = params['maximum_cost']
        self.inner_weight = params['inner']
        self.outer_weight = params['outer']
        self.local_weight = params['local']

        self.n_bins = n_bins
        self.smooth_sigma = std
        self.length = train_segment_size

        self.local_feats = local_feats
        self.inner_feats = inner_feats
        self.outer_feats = outer_feats

    def process_series(self, series):
        hist = np.zeros(self.length)

        for i, idx in enumerate(series):
            hist[self.inner_feats(idx)] += 1

        hist = gaussian_filter1d(hist, self.smooth_sigma)
        feats = np.ceil(self.maximum_cost * self.local_feats * (1 - hist / np.max(hist)))
        return hist
