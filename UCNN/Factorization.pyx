import numpy as np
cimport numpy as np

cpdef conv_factorization(np.ndarray kernel):
    """
        It gets the kernel and returns a dictionary with shape(W, indexes) where W is the
        weight and "indexes" is the indexes where W happens.
    :param kernel:
    :return:
    repeated_dict: A dictionary of type: (key, value) where "key"s are the weights, and "value"s are
    indexes of unique weight
    max_index: The weight whose repeated indexes are most
    min_index: The weight whose repeated indexes are least
    """
    cdef:
        np.ndarray weight = np.unique(kernel)
        np.ndarray indices = np.zeros(len(weight), dtype=int)
        np.ndarray repeated = np.empty([0, 3], dtype=int)
        float max = 0
        float min = np.inf
        int max_index = 0
        int min_index = 0
        int i = 0
        np.ndarray index
        tuple temp
    while i < weight.shape[0]:
        temp = np.where(kernel == weight[i])

        index = np.concatenate((temp[0][np.newaxis], temp[1][np.newaxis], temp[2][np.newaxis],), axis=0).T
        repeated = np.append(repeated, index, axis=0)
        indices[i] = len(index) + indices[i - 1] if i > 0 else len(index)
        i += 1

    return weight, repeated, indices
