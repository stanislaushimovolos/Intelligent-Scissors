import numpy as np
from dijkstar import Graph
from itertools import product
from skimage.filters import laplace, sobel_h, sobel_v

from .utils import unfold

shifts = [(-1, 1), (-1, 0), (-1, -1), (0, 1), (0, -1), (1, -1), (1, 0), (1, 1)]

default_weights = {
    'laplace': 0,
    'gradient_direction': 0.9,
    'gradient_magnitude': 0.1
}


# TODO replace function by its methods
class FeatureExtractor:
    def __init__(self):
        pass


def get_gradient_magnitude_feats(image):
    grads = np.array([sobel_h(image), sobel_v(image)])
    grads = np.transpose(grads, (1, 2, 0))
    norm = np.linalg.norm(grads, axis=2)
    feats = 1 - norm / np.max(norm)
    return feats


# TODO that's wrong
def get_laplacian_feats(img, ksize):
    laplacian = laplace(img, ksize=ksize)
    return laplacian < 0.1


def get_reversed_sobel(img):
    sobel = np.stack([-sobel_h(img), sobel_v(img)])
    return sobel


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


def get_gradient_direction_feats(img):
    filter_size = np.array([3, 3])
    feature_size = 2

    sobel = get_reversed_sobel(img)
    unfolded_sobel, _ = unfold(sobel, filter_size)
    spatial_feats = create_spatial_feats(img.shape, filter_size)

    sign_mask = np.sign(np.sum(spatial_feats * np.expand_dims(np.expand_dims(sobel, 1), 1), axis=0))

    distant_feats = sign_mask * np.sum(spatial_feats * unfolded_sobel, axis=0)
    local_feats = sign_mask * np.sum(np.expand_dims(np.expand_dims(sobel, 1), 1) * spatial_feats, axis=0)

    total_cost = 2 / (3 * np.pi) * (np.arccos(local_feats) + np.arcsin(distant_feats))
    return total_cost


def get_total_link_costs(img, weights=None):
    if weights is None:
        weights = default_weights

    laplace_w = weights['laplace']
    magnitude_w = weights['gradient_magnitude']
    direction_w = weights['gradient_direction']

    # TODO remove magic numbers
    direction_cost = get_gradient_direction_feats(img)
    laplacian_fets = np.expand_dims(get_laplacian_feats(img, 5), 0)
    laplacian_feats, _ = unfold(laplacian_fets, np.array([3, 3]))

    gradient_fets = np.expand_dims(get_gradient_magnitude_feats(img), 0)
    gradient_fets, _ = unfold(gradient_fets, np.array([3, 3]))

    total_cost = laplace_w * laplacian_fets + magnitude_w * gradient_fets + direction_w * direction_cost
    return total_cost


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
