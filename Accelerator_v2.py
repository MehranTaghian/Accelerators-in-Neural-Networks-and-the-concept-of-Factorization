import numpy as np
from Accelerator import factorization

def conv_factorization(kernel):
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
    weights = np.unique(kernel)
    # print(len(weights))
    list_indexes_size = np.zeros(len(weights), dtype=int)
    repeated_indexes = np.zeros([np.size(kernel), 3], dtype=int)
    counter = 0
    for i in weights:
        index = np.where(kernel == i)
        list_indexes_size[counter] = len(index[0]) + list_indexes_size[counter - 1] if counter != 0 else len(index[0])
        # print(np.concatenate((index[0][:, np.newaxis], index[1][:, np.newaxis], index[2][:, np.newaxis]), axis=1))
        repeated_indexes[list_indexes_size[counter - 1] if counter > 0 else 0: list_indexes_size[counter]] = \
            np.concatenate((index[0][:, np.newaxis], index[1][:, np.newaxis], index[2][:, np.newaxis]), axis=1)

        counter += 1
    list_indexes_size = list_indexes_size[0:len(list_indexes_size) - 1]
    return repeated_indexes, list_indexes_size, weights


def conv2d(data, kernel, repeated_position, weights, indexes, stride=1):
    """
        The kernel is 3d like in image with 3 channels RGB
    :param data: is the actual input to convolution
    :param kernel: is the kernel of convolution e.g. a 3 by 3 kernel
    :param repeated_position: is the position of repeated weights
    :param stride: steps of moving kernel
    :return: the result of convolution
    """
    result_size = int((data.shape[0] - kernel.shape[0]) / stride + 1)
    result = np.zeros([result_size, result_size])
    for i in range(0, int((data.shape[0] - kernel.shape[0]) / stride) + 1, stride):
        for j in range(0, int((data.shape[1] - kernel.shape[1]) / stride) + 1, stride):
            weights_size = len(weights)
            # print(repeated_position[np.arange(weights_size)][0],
            #       repeated_position[np.arange(weights_size)][1],
            #       repeated_position[np.arange(weights_size)][2])

            # print(repeated_position[np.arange(weights_size)][np.arange(weights_size)])

            result[i, j] = np.sum(np.concatenate(weights * \
                                  np.array(np.split(data[repeated_position[:, 0] + i,
                                                         repeated_position[:, 1] + j, repeated_position[:, 2]], indexes),
                                           dtype=object)))
            # temp = weights[np.arange(weights_size)] * data[repeated_position[np.arange(weights_size)][0],
            #                                                repeated_position[np.arange(weights_size)][1],
            #                                                repeated_position[np.arange(weights_size)][2]]

            # result[i, j] = np.sum(temp)
    return result


a = np.random.randint(5, size=(5, 5, 5))
k = np.random.randint(5, size=(3, 3, 5))
# k = np.array([
#     [[1, 2, 3],
#      [2, 3, 4],
#      [6, 7, 8]],
#     [[1, 5, 6],
#      [2, 3, 5],
#      [4, 5, 6]],
#     [[4, 5, 6],
#      [4, 6, 7],
#      [1, 8, 9]]])

repeated, indexes, weights = conv_factorization(k)
print(indexes)
print(weights)
# print(indexes[np.arange(len(indexes) - 1)])
# print(indexes[np.arange(1, len(indexes))])
r = conv2d(a, k, repeated, weights, indexes)
print(r)

temp, _, _ = factorization.conv_factorization(k)
r2, _, _ = factorization.conv2d(a, k, temp)
print(r2)
