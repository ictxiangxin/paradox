import numpy


def generate_label_matrix(classification, default: int=0):
    index_map = {}
    index_reverse_map = {}
    index = 0
    for c in classification:
        if c not in index_map:
            index_map[c] = index
            index_reverse_map[index] = c
            index += 1
    dimension = len(index_map)
    class_matrix = default * numpy.ones([len(classification), dimension])
    for i, c in enumerate(classification):
        class_matrix[i, index_map[c]] = 1
    return class_matrix, index_map, index_reverse_map
