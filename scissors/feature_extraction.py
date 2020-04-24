import numpy as np
from skimage.filters import gaussian, laplace, sobel_h, sobel_v

from scissors.utils import unfold, create_spatial_feats, flatten_first_dims, norm_by_max

default_weights = {
    'laplace': 0.4,
    'direction': 0.2,
    'magnitude': 0.4
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

    def __call__(self, image):
        return self.get_total_link_costs(image)

    def get_total_link_costs(self, image):
        l_cost = self.get_laplace_cost(image, self.laplace_filter_size, self.gaussian_std)
        l_cost = unfold(l_cost, self.filter_size)

        g_cost = self.get_magnitude_cost(image)
        g_cost = unfold(g_cost, self.filter_size)

        d_cost = self.get_direction_cost(image)
        total_cost = self.laplace_w * l_cost + self.magnitude_w * g_cost + self.direction_w * d_cost
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

    def get_direction_cost(self, img):
        grads = np.stack([-sobel_h(img), sobel_v(img)])
        unfolded_grads = unfold(grads, self.filter_size)
        grads = grads[:, None, None, ...]

        spatial_feats = create_spatial_feats(img.shape, self.filter_size)
        link_feats = np.einsum('i..., i...', spatial_feats, grads)
        local_feats = np.abs(link_feats)

        sign_mask = np.sign(link_feats)
        distant_feats = sign_mask * np.einsum('i..., i...', spatial_feats, unfolded_grads)
        total_cost = 2 / (3 * np.pi) * (np.arccos(local_feats) + np.arcsin(distant_feats))
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

        self.inner_feats = self.get_inner_feats(unfolded_feats, dots_products)
        self.outer_feats = self.get_outer_feats(unfolded_feats, dots_products)

    @staticmethod
    def get_inner_feats(feats, dots_products):
        indices = np.argmax(dots_products, axis=0).astype(np.int)
        return np.choose(indices, feats)

    @staticmethod
    def get_outer_feats(feats, dots_products):
        indices = np.argmin(dots_products, axis=0).astype(np.int)
        return np.choose(indices, feats)
