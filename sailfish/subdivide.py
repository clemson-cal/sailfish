"""
Utility functions for domain decomposition.
"""


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
