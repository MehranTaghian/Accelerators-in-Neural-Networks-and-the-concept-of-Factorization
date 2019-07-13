import numpy as np
import tensorflow as tf


def quantization(kernel):
    r_max = np.max(kernel)
    r_min = np.min(kernel)
    quantized = np.int8(255 / (r_max - r_min) * kernel)
    return quantized


def convolve(input_data, conv_layer, bias, layer_num, padding="VALID", stride=1):
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
        kernel = conv_layer[:, :, :, f]
        repeated_w = conv_factorization(kernel)
        conv_result = conv2d(input_data, kernel, repeated_w, stride)
        result[:, :, f] = conv_result + bias[f]

        # Experimental part
        # -----------------------------
        print("Filter", f)
        common = get_common_regions(input_data, kernel, repeated_w, stride)
        write_common_regions_csv(common, 'layer-' + layer_num, f)

        # print("Maximum repeated weight:", max_index, "Number of repeated:", len(repeated_w[max_index][0]))
        # print("Minimum repeated weight: ", min_index, "Number of repeated:", len(repeated_w[min_index][0]))

        # print("Number of sum", number_of_sum)
        # print("Number of prod", number_of_prod)

        # -----------------------------------------

    return result


def write_common_regions_csv(common, layer_name, filter_num):
    import csv
    csv.register_dialect('myDialect', delimiter='|', quoting=csv.QUOTE_ALL)
    if filter_num == 0:
        with open('common-' + layer_name + '.csv', 'w') as csvfile:
            fieldnames = ['weights', 'positions']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect="myDialect")
            writer.writeheader()
            for i in common:
                key = i + ('Filter' + str(filter_num + 1),)
                writer.writerow({'weights': key, 'positions': common[i]})
    else:
        with open('common-' + layer_name + '.csv', 'a') as csvfile:
            fieldnames = ['weights', 'positions']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for i in common:
                key = i + ('Filter' + str(filter_num + 1),)
                writer.writerow({'weights': key, 'positions': common[i]})

    print("writing completed")


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

        # Experimental part
        if len(index[0]) > max:
            max = len(index[0])
            max_index = i
        if len(index[0]) < min:
            min = len(index[0])
            min_index = i
        # ----------------------------------

        repeated_dict[i] = index
    return repeated_dict  # , max_index, min_index


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
    number_of_sum = np.zeros(4)
    number_of_prod = np.zeros(4)
    number_of_memory_access = np.zeros(4)
    size_of_zero_weights = 0
    weights_inputs_common = {}
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
                    # Experimental part
                    # mode4
                    number_of_sum[3] += len(repeated_position[ind][0])
                    number_of_prod[3] += 1
                    number_of_memory_access[3] += (1  # reading weight
                                                   +
                                                   len(repeated_position[ind][
                                                           0]))  # reading positions in input for that weight
                    # -----------------------------------
                else:
                    size_of_zero_weights = len(repeated_position[ind][0])

                # Experimental part
                if i == j == 0:
                    weights_inputs_common[ind] = np.empty([0, 3])
                weights_inputs_common[ind] = np.append(weights_inputs_common[ind], [repeated_position[ind][0] + i,
                                                                                    repeated_position[ind][1] + j,
                                                                                    repeated_position[ind][2]])

            # mode1
            number_of_sum[0] += np.size(kernel)
            number_of_prod[0] += np.size(kernel)
            number_of_memory_access[0] += 2 * np.size(kernel)

            # mode2
            number_of_sum[1] += (np.size(kernel) - size_of_zero_weights)
            number_of_prod[1] += (np.size(kernel) - size_of_zero_weights)

            # we read zero weights only for understanding that they are zero.
            # As soon as we understood, we ignore input data of that position
            number_of_memory_access[1] += 2 * (np.size(kernel) - size_of_zero_weights) + size_of_zero_weights

            # mode3
            # number_of_sum[3] +=
            # -----------------------------------

            result[i, j] = temp_result
    return result, weights_inputs_common  # , number_of_sum, number_of_prod


def get_common_regions(data, kernel, repeated_position, stride=1):
    common = {}
    for i in range(0, int((data.shape[0] - kernel.shape[0]) / stride) + 1, stride):
        for j in range(0, int((data.shape[1] - kernel.shape[1]) / stride) + 1, stride):

            for w1 in repeated_position:
                for w2 in repeated_position:
                    if w1 != w2 and ((w2, w1) not in common.keys()):
                        common[(w1, w2)] = []
                        data1 = data[repeated_position[w1][0] + i, repeated_position[w1][1] + j,
                                     repeated_position[w1][2]]
                        data2 = data[repeated_position[w2][0] + i, repeated_position[w2][1] + j,
                                     repeated_position[w2][2]]

                        for x1 in range(len(data1)):
                            for x2 in range(len(data2)):
                                if data1[x1] == data2[x2]:
                                    common[(w1, w2)].append((
                                        (repeated_position[w1][0][x1] + i, repeated_position[w1][1][x1] + j,
                                         repeated_position[w1][2][x1]),
                                        (repeated_position[w2][0][x2] + i, repeated_position[w2][1][x2] + j,
                                         repeated_position[w2][2][x2])))

    return common

# Experimental part


# a = np.ones([5, 5, 5])


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

# a = np.random.randint(5, size=(5, 5, 5))
# k = np.random.randint(5, size=(3, 3, 5, 3))

# a = np.ones([5, 5, 5])


# k = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9]])


# k = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9]])


# temp = conv_factorization(k)
# r = conv2d(a, k, temp)
# r = convolve(a, k, np.zeros([k.shape[3]]), 'conv1')
# print(r)
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
