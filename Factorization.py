import numpy as np
import tensorflow as tf

# a = np.array([
#     [[1, 2, 3],
#      [2, 3, 4],
#      [6, 7, 8]],
#     [[1, 5, 6],
#      [2, 3, 5],
#      [4, 5, 6]],
#     [[4, 5, 6],
#      [4, 6, 7],
#      [1, 8, 9]]])


a = np.random.randint(5, size=(5, 5, 5))
k = np.random.randint(5, size=(3, 3, 5))


# a = np.ones([5, 5, 5])


# k = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9]])


def conv_factorization(kernel):
    """
        It gets the kernel and returns a dictionary with shape(W, indexes) where W is the
        weight and "indexes" is the indexes where W happens.
    :param kernel:
    :return:
    """
    uniq = np.unique(kernel)
    repeated_dict = []
    for i in uniq:
        index = np.where(kernel == i)
        repeated_dict.append((i, index))
    return repeated_dict


def conv2d(data, kernel, repeated_position, stride=1):
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
    for i in range(0, data.shape[0] - kernel.shape[0] + 1, stride):
        for j in range(0, data.shape[1] - kernel.shape[1] + 1, stride):
            temp_result = 0
            for ind in range(len(repeated_position)):
                """ 
                    repeated_position[ind][0]: is the weight
                    repeated_position[ind][1]: is the indexes
                """
                temp_result += repeated_position[ind][0] * np.sum(data[repeated_position[ind][1][0] + i,
                                                                       repeated_position[ind][1][1] + j,
                                                                       repeated_position[ind][1][2]])
            result[i, j] = temp_result
    return result


temp = conv_factorization(k)
r = conv2d(a, k, temp)
print(r)

# tensor_a = tf.constant(a, tf.float32)
# tensor_k = tf.constant(k, tf.float32)
#
# tensor_res = tf.nn.convolution(tf.reshape(tensor_a, [1, 5, 5, 5]), tf.reshape(tensor_k, [3, 3, 5, 1]), padding='VALID')
#
# sess = tf.Session()
# print(sess.run(tensor_res))

# print(sess.run(tf.reshape(tensor_a, [1, 5, 5, 5])))
# print(np.sum(a))





# print(temp[0][1])

# for index in range(len(temp)):
#     # print(temp[index][1][0])
#     # print(temp[index][1][1])
#     # print(temp[index][1][2])
#     print('len', len(temp[index][1]))
#     print(np.sum(a[temp[index][1][0], temp[index][1][1], temp[index][1][2]]))

# print(np.sum(a[temp[np.arange(len(temp))][1][0], temp[np.arange(len(temp))][1][1], temp[np.arange(len(temp))][1][2]]))


# b = np.unique(a, return_index=True)
#
# temp = np.where(a == b[0][0])
# index = (temp[0], temp[1])
# print(index)
