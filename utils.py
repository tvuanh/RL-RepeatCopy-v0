from __future__ import division, print_function, absolute_import


def encode_action(action):
    """
    >>> encode_action((0, 0, 0))
    0
    >>> encode_action((1, 0, 0))
    1
    >>> encode_action((1, 1, 0))
    3
    >>> encode_action((1, 1, 1))
    7
    >>> encode_action((1, 1, 4))
    19
    >>> encode_action((1, 0, 4))
    17
    """
    first, second, third = action
    return first + 2 * second + 4 * third


def decode_action(action):
    """
    >>> decode_action(0)
    (0, 0, 0)
    >>> decode_action(1)
    (1, 0, 0)
    >>> decode_action(2)
    (0, 1, 0)
    >>> decode_action(4)
    (0, 0, 1)
    >>> decode_action(9)
    (1, 0, 2)
    """
    third = action // 4
    second = (action - 4 * third) // 2
    first = action - 4 * third - 2 * second
    return (first, second, third)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
