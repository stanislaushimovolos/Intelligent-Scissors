import numpy as np
from itertools import product


def iterate_by_shift():
    pass


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

    return unfolded, shift_boundaries
