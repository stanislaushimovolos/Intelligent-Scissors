import numpy as np
from itertools import product


def unfold(x, filter_size):
    feature_size, *spatial = x.shape
    unfolded = np.zeros((feature_size, *filter_size, *spatial))

    def get_spans(shift):
        if shift > 0:
            source_span = slice(0, -shift)
            shifted_span = slice(shift, None)
        elif shift < 0:
            source_span = slice(-shift, None)
            shifted_span = slice(0, shift)
        else:
            shifted_span = source_span = slice(0, None)
        return source_span, shifted_span

    start_span_coord = filter_size // 2
    stop_span_coord = filter_size - start_span_coord - 1
    shift_boundaries = [
        np.arange(-start_coord, stop_coord + 1)
        for start_coord, stop_coord in zip(start_span_coord, stop_span_coord)
    ]

    for shifts in product(*shift_boundaries):
        cur_source_span, cur_shifted_span = zip(*map(get_spans, shifts))
        current_slice = (...,) + tuple(shifts + start_span_coord) + cur_source_span
        unfolded[current_slice] = x[(...,) + cur_shifted_span]

    return unfolded


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


def flatten_first_dims(x, n_dims=2):
    shape = x.shape
    return np.reshape(x, ((np.product(shape[:n_dims]),) + shape[n_dims:]))


def norm_by_max_value(feats, max_val):
    feats = feats / np.max(feats)
    return max_val * feats


def get_static_cost(u, v, edge, prev_edge):
    return edge


def quadratic_kernel(x, size):
    return 1 - (x / size) ** 2
