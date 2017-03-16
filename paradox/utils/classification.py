import numpy


def generate_class_matrix(classification, default: int=0):
    index_map = {}
    index = 0
    for c in classification:
        if c not in index_map:
            index_map[c] = index
            index += 1
    dimension = len(index_map)
    class_matrix = default * numpy.ones([dimension, len(classification)])
    for i, c in enumerate(classification):
        class_matrix[index_map[c], i] = 1
    return class_matrix
