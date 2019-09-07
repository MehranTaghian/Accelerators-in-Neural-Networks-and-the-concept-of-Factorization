import numpy as np
from PE import PE
import tensorflow as tf


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
    weight = np.unique(kernel)
    indices = np.zeros(len(weight), dtype=int)
    repeated = np.empty([0, 3], dtype=int)
    i = 0
    while i < weight.shape[0]:
        temp = np.where(kernel == weight[i])
        index = np.concatenate((temp[0][np.newaxis], temp[1][np.newaxis], temp[2][np.newaxis],), axis=0).T
        repeated = np.append(repeated, index, axis=0)
        indices[i] = len(index) + indices[i - 1] if i > 0 else len(index)
        i += 1

    indices = indices[:len(indices) - 1]
    return weight, np.array(np.split(repeated, indices), dtype=object) if len(indices) > 0 else repeated


class CU:
    def __init__(self, inputs, kernel):
        self.unique_weights, repeated = conv_factorization(kernel)
        self.inputs_for_unique_weights = {}
        for i in range(len(self.unique_weights)):
            self.inputs_for_unique_weights[self.unique_weights[i]] = inputs[
                repeated[i][:, 0], repeated[i][:, 1], repeated[i][:, 2]]


def conv_single_stride(inputs, filter):
    cu = CU(inputs, filter)
    global_8bit_adder = 0

    for weight in cu.unique_weights:
        pe = PE(weight)
        i = 0
        while i < len(cu.inputs_for_unique_weights[weight]):
            if i + 8 <= len(cu.inputs_for_unique_weights[weight]):
                pe.parallel_8bit_adder(cu.inputs_for_unique_weights[weight][i:i + 8])
                i += 8
            else:
                pe.parallel_8bit_adder(
                    cu.inputs_for_unique_weights[weight][i: len(cu.inputs_for_unique_weights[weight])],
                    len(cu.inputs_for_unique_weights[weight]) - i)
                break
        global_8bit_adder += pe.psum * weight

    return global_8bit_adder


def conv2d(data, filter, stride=1):
    """
        The kernel is 3d like in image with 3 channels RGB
    :param filter:
    :param filter2:
    :param data: is the actual input to convolution
    :param stride: steps of moving kernel
    :return: the result of convolution
    """

    result_size = int((data.shape[0] - filter.shape[0]) / stride + 1)
    result = np.zeros([result_size, result_size])
    i = 0

    while i < (result.shape[0]):
        j = 0
        while j < (result.shape[1]):
            row_index, col_index = i * stride, j * stride
            result[i, j] = conv_single_stride(
                data[row_index: row_index + filter.shape[0], col_index: col_index + filter.shape[1], :], filter)

            j += 1
        i += 1

    return result


def convolution(data, kernel, bias, layer_num, stride=1, padding="VALID"):
    input_data_padded = data
    if padding == "SAME":
        p = int((kernel.shape[0] - 1) / 2)
        input_data_padded = np.zeros([data.shape[0] + 2 * p, data.shape[1] + 2 * p, data.shape[2]])
        for i in range(data.shape[2]):
            input_data_padded[:, :, i] = np.pad(data[:, :, i], (p, p), mode="constant")
    filter_num = kernel.shape[3]
    output_width = int((input_data_padded.shape[0] - kernel.shape[0]) / stride + 1)
    output_size = [output_width, output_width, filter_num]
    result = np.zeros(output_size)
    # workbook = excel.Workbook(f'result\\{layer_num}.xlsx')
    # worksheet = workbook.add_worksheet(layer_num)
    for i in range(0, filter_num):
        result[:, :, i] = conv2d(input_data_padded,
                                 kernel[:, :, :, i],
                                 stride)

    # workbook.close()
    return result + bias








# a = np.random.randint(0, 5, size=(29, 29, 3))
# k = np.random.randint(0, 5, size=(5, 5, 3, 30))
# result1 = convolution(a, k, 0, 'conv1', stride=4, padding='SAME')
#
# tensor_a = tf.constant(a, tf.float32)
# tensor_k = tf.constant(k, tf.float32)
# sess = tf.Session()
#
# # tensor_res = tf.nn.convolution(tf.reshape(tensor_a, [1, 29, 29, 3]), tf.reshape(tensor_k, [3, 3, 3, 2]), padding='VALID')
# tensor_res = tf.nn.convolution(tf.reshape(tensor_a, [1, 29, 29, 3]), tf.reshape(tensor_k, [5, 5, 3, 30]),
#                                strides=[1, 4, 4, 1], padding='SAME')
#
# result2 = sess.run(tensor_res)
# print((result1 == result2).all())
