import numpy as np


from memory_profiler import profile


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


def quantization(kernel):
    r_max = np.max(kernel)
    r_min = np.min(kernel)
    quantized = np.int8(255 / (r_max - r_min) * kernel)
    return quantized


def convolve(input_data, conv_layer, bias, padding="VALID", stride=1):
    """

    :param input_data: input data to the kernel
    :param conv_layer: 4D convolution layer to be convolved into input_data
    :param bias:
    :param padding: if "SAME", we add padding to the input data so that
                    the output would be the same size as input
    :param stride:
    :return: result of convolution
    """
    if padding == "SAME":
        p = int((conv_layer.shape[0] - 1) / 2)
        input_data = np.pad(input_data, p, mode="constant")

    filter_num = conv_layer.shape[3]
    output_width = int((input_data.shape[0] - conv_layer.shape[0]) / stride + 1)
    output_size = [output_width, output_width, filter_num]
    result = np.zeros(output_size)

    conv_layer = quantization(conv_layer)
    for f in range(filter_num):
        print("Filter", f)
        kernel = conv_layer[:, :, :, f]
        repeated_w, max_index, min_index = conv_factorization(kernel)
        # print("Maximum repeated weight:", max_index, "Number of repeated:", len(repeated_w[max_index][0]))
        # print("Minimum repeated weight: ", min_index, "Number of repeated:", len(repeated_w[min_index][0]))
        conv_result, number_of_sum, number_of_prod = conv2d(input_data, kernel, repeated_w, stride)
        result[:, :, f] = conv_result + bias[f]
        print("Number of sum", number_of_sum)
        print("Number of prod", number_of_prod)

    return result


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
    uniq = np.unique(kernel)
    repeated_dict = {}
    max = 0
    min = np.inf
    max_index = min_index = 0
    for i in uniq:
        index = np.where(kernel == i)
        if len(index[0]) > max:
            max = len(index[0])
            max_index = i
        if len(index[0]) < min:
            min = len(index[0])
            min_index = i
        repeated_dict[i] = index
    return repeated_dict, max_index, min_index


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
    number_of_sum = 0
    number_of_prod = 0

    for i in range(0, int((data.shape[0] - kernel.shape[0]) / stride) + 1, stride):
        for j in range(0, int((data.shape[1] - kernel.shape[1]) / stride) + 1, stride):
            temp_result = 0
            for ind in repeated_position:
                """ 
                    repeated_position[ind][0]: is the weight
                    repeated_position[ind][1]: is the indexes
                """
                if ind != 0:  # Zero weights
                    temp_result += ind * np.sum(data[repeated_position[ind][0] + i,
                                                     repeated_position[ind][1] + j,
                                                     repeated_position[ind][2]])
                    number_of_sum += len(repeated_position[ind][0])
                    number_of_prod += 1
            result[i, j] = temp_result
    return result, number_of_sum, number_of_prod

temp, _, _ = conv_factorization(k)
r, _, _ = conv2d(a, k, temp)
print(r)
# import tensorflow as tf
#
# tensor_a = tf.constant(a, tf.float32)
# tensor_k = tf.constant(k, tf.float32)
# #
# tensor_res = tf.nn.convolution(tf.reshape(tensor_a, [1, 5, 5, 5]), tf.reshape(tensor_k, [3, 3, 5, 1]), padding='VALID')
# #
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
