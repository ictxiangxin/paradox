from functools import reduce


def array_index_traversal(array_shape):
    scale = reduce(lambda a, b: a * b, array_shape)
    for i in range(scale):
        index = [0] * len(array_shape)
        index[-1] = i
        for r in range(len(array_shape) - 1, -1, -1):
            if index[r] >= array_shape[r]:
                index[r - 1] = index[r] // array_shape[r]
                index[r] %= array_shape[r]
            else:
                break
        yield tuple(index)


def multi_range(range_list):
    len_list = [len(r) for r in range_list]
    scale = reduce(lambda a, b: a * b, len_list)
    for i in range(scale):
        index = [0] * len(len_list)
        index[-1] = i
        for r in range(len(len_list) - 1, -1, -1):
            if index[r] >= len_list[r]:
                index[r - 1] = index[r] // len_list[r]
                index[r] %= len_list[r]
            else:
                break
        yield tuple(range_list[_i][_index] for _i, _index in enumerate(index))
