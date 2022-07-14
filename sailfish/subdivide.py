"""
Utility functions for domain decomposition.
"""


def to_host(a):
    try:
        return a.get()
    except AttributeError:
        return a


def lazy_reduce(reduction, block, launches, contexts):
    """
    Applies a reduction over a sequence of parallelizable device operations.

    The reduction can be something like built-in `max` or `min`. The
    `launches` argument is a sequence of callables which trigger the
    asynchronous calculation and return a "token" (in the case of a cupy
    reduction, the token is a device array). The `block` argument is a
    callable which operates on the token, blocking until the underlying
    operation is completed, and then returns a value (`block` is probably
    built-in `float`). The `contexts` argument is a sequence of execution
    contexts which should switch to the device on which the respective launch
    callable was executed.
    """
    tokens = [launch() for launch in launches]
    results = []
    for token, context in zip(tokens, contexts):
        with context:
            results.append(block(token))
    return reduction(results)


def partition(elements, num_parts):
    """
    Equitably divide the given number of elements into `num_parts` partitions.

    The sum of the partitions is `elements`. The number of partitions must be
    less than or equal to the number of elements.
    """
    n = elements // num_parts
    r = elements % num_parts

    for i in range(num_parts):
        yield n + (1 if i < r else 0)


def subdivide(interval, num_parts):
    """
    Divide an interval into non-overlapping contiguous sub-intervals.
    """
    try:
        a, b = interval
    except TypeError:
        a, b = 0, interval

    for n in partition(b - a, num_parts):
        yield a, a + n
        a += n


def concat_on_host(arrays: list, num_guard=None, rank=None):
    """
    Concatenate a list of arrays, which may be allocated on different devices.

    The array returned is allocated on the host. The concatenation is
    performed on the first axis.

    Arrays represent fields of data on either 1d or 2d grids (rank of either 1
    or 2). If rank is None, it is assumed that only the final axis contains
    fields, so rank is inferred to be len(array.shape) - 1. If several
    trailing axes represent more fields, then rank must be given explicitly.
    """
    import numpy as np

    def all_equal(seq):
        for x in seq:
            try:
                if x != y:
                    raise ValueError("got distinct values")
            except UnboundLocalError:
                y = x
        return x

    if rank is None:
        rank = len(arrays[0].shape) - 1

    if rank == 1:
        # TODO: 1d case
        ng = num_guard or 0
        ni = sum(a.shape[0] - 2 * ng for a in arrays)
        nq = all_equal(a.shape[1:] for a in arrays)
        si = slice(ng, -ng) if ng > 0 else slice(None)
        result = np.zeros((ni,) + nq)
        i = 0
        for array in arrays:
            a = i
            b = i + array.shape[0] - 2 * ng
            i += b - a
            result[a:b] = to_host(array[si])
        return result

    if rank == 2:
        ngi, ngj = num_guard or (0, 0)
        ni = sum(a.shape[0] - 2 * ngi for a in arrays)
        nj = all_equal(a.shape[1] - 2 * ngj for a in arrays)
        nq = all_equal(a.shape[2:] for a in arrays)
        si = slice(ngi, -ngi) if ngi > 0 else slice(None)
        sj = slice(ngj, -ngj) if ngj > 0 else slice(None)
        result = np.zeros((ni, nj) + nq)

        i = 0
        for array in arrays:
            a = i
            b = i + array.shape[0] - 2 * ngi
            i += b - a
            result[a:b] = to_host(array[si, sj])

        return result

    raise ValueError(f"concatenation for arrays of rank {rank} not supported")
