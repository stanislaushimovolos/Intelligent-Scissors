import skimage
import numpy as np
from typing import Sequence, Union
from itertools import product


def unfold(x: np.array, filter_size: Union[int, np.array] = 3):
    feature_size, *spatial = x.shape
    if isinstance(filter_size, int):
        filter_size = np.array((filter_size,) * len(spatial))

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


def create_spatial_feats(shape: Sequence[int], filter_size: Union[int, np.array] = 3, feature_size: int = 2):
    if isinstance(filter_size, int):
        filter_size = np.array((filter_size,) * len(shape))

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


def preprocess_image(image: np.array):
    if len(image.shape) == 2:
        brightness = image
        image = np.expand_dims(image, 0)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        brightness = skimage.color.rgb2hsv(image)[:, :, 2]
        image = np.transpose(image, (2, 0, 1))
    else:
        raise RuntimeError
    return image, brightness


def flatten_first_dims(x, n_dims=2):
    shape = x.shape
    return np.reshape(x, ((np.product(shape[:n_dims]),) + shape[n_dims:]))


def quadratic_kernel(x, size):
    return 1 - (x / size) ** 2
