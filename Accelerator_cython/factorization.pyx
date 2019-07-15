import tensorflow as tf
import numpy as np
cimport numpy as np
cimport cython
cimport cython

cdef np.ndarray quantization(np.ndarray kernel):
    cdef:
        np.npy_float32 r_max = np.max(kernel)
        np.npy_float32 r_min = np.min(kernel)
        np.ndarray quantized = np.int8(255 / (r_max - r_min) * kernel)
    return quantized

@cython.cdivision(True)
cpdef np.ndarray convolve(np.ndarray input_data, np.ndarray conv_layer, np.ndarray bias, str layer_num,
                          str padding="VALID", int stride=1):
    """

    :param layer_num: 
    :param input_data: input data to the kernel
    :param conv_layer: 4D convolution layer to be convolved into input_data
    :param bias:
    :param padding: if "SAME", we add padding to the input data so that
                    the output would be the same size as input
    :param stride:
    :return: result of convolution
    """
    cdef int p = 0
    cdef np.ndarray input_data_padded = input_data
    if padding == "SAME":
        p = int((conv_layer.shape[0] - 1) / 2)
        input_data_padded = np.pad(input_data, p, mode="constant")
    cdef int filter_num = conv_layer.shape[3]
    cdef int output_width = int((input_data_padded.shape[0] - conv_layer.shape[0]) / stride + 1)
    cdef list output_size = [output_width, output_width, filter_num]
    cdef np.ndarray result = np.zeros(output_size)

    cdef np.ndarray kernel, conv_result
    cdef dict repeated_w, weights_inputs_common
    cdef np.ndarray conv_layer_quantized = quantization(conv_layer)

    for f in range(filter_num):
        kernel = conv_layer_quantized[:, :, :, f]
        # kernel = conv_layer[:, :, :, f]

        print('factoring')
        repeated_w = conv_factorization(kernel)
        print('convolving')
        conv_result, weights_inputs_common = conv2d(input_data_padded, kernel, repeated_w, stride)
        result[:, :, f] = conv_result + bias[f]
        # Experimental part
        # -----------------------------
        print("Filter", f)
        # common = get_common_regions(weights_inputs_common)
        # write_common_regions_csv(common, 'layer-' + layer_num, f)

        # print("Maximum repeated weight:", max_index, "Number of repeated:", len(repeated_w[max_index][0]))
        # print("Minimum repeated weight: ", min_index, "Number of repeated:", len(repeated_w[min_index][0]))

        # print("Number of sum", number_of_sum)
        # print("Number of prod", number_of_prod)

        # -----------------------------------------

    return result

def write_common_regions_csv(common, layer_name, filter_num):
    import csv
    csv.register_dialect('myDialect', delimiter=':', quoting=csv.QUOTE_ALL)
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

cdef dict conv_factorization(np.ndarray kernel):
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
        np.ndarray uniq = np.unique(kernel)
        dict repeated_dict = {}
        float max = 0
        float min = np.inf
        int max_index = 0
        int min_index = 0
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

@cython.cdivision(True)
cdef conv2d(np.ndarray data, np.ndarray kernel, dict repeated_position, int stride=1):
    """
        The kernel is 3d like in image with 3 channels RGB
    :param data: is the actual input to convolution
    :param kernel: is the kernel of convolution e.g. a 3 by 3 kernel
    :param repeated_position: is the position of repeated weights
    :param stride: steps of moving kernel
    :return: the result of convolution
    """
    cdef:
        int result_size = int((data.shape[0] - kernel.shape[0]) / stride + 1)
        np.ndarray result = np.zeros([result_size, result_size])
        int number_of_sum[4], number_of_prod[4], number_of_memory_access[4]
        int size_of_zero_weights = 0
        dict weights_inputs_common = {}
        int temp_result
        int ind
        int width = ((data.shape[1] - kernel.shape[1]) / stride) + 1
        int height = ((data.shape[0] - kernel.shape[0]) / stride) + 1
        int i = 0, j = 0

    while i < height:
        j = 0
        while j < width:
            temp_result = 0
            for ind in repeated_position:
                pass
                #     """
                #         repeated_position[ind][0]: is the weight
                #         repeated_position[ind][1]: is the indexes
                #     """
                if ind != 0:  # Zero weights
                    multiply(data, repeated_position[ind], ind)

                    # temp_result += ind * np.sum(data[repeated_position[ind][0] + i,
                    #                                  repeated_position[ind][1] + j,
                    #                                  repeated_position[ind][2]])
                    # Experimental part
                    # mode4
                    # number_of_sum[3] += len(repeated_position[ind][0])
                    # number_of_prod[3] += 1
                    # number_of_memory_access[3] += (1  # reading weight
                    #                                +
                    #                                len(repeated_position[ind][
                    #                                        0]))  # reading positions in input for that weight
                    # -----------------------------------
                # else:
                #     size_of_zero_weights = len(repeated_position[ind][0])

                # Experimental part
                # if i == j == 0:
                #     weights_inputs_common[ind] = np.empty([0, 3])
                #
                # weights_inputs_common[ind] = np.append(weights_inputs_common[ind],
                #                                        np.concatenate(((repeated_position[ind][0] + i)[np.newaxis],
                #                                                        (repeated_position[ind][1] + j)[np.newaxis],
                #                                                        (repeated_position[ind][2])[np.newaxis]),
                #                                                       axis=0).T, axis=0)

            # mode1
            # number_of_sum[0] += np.size(kernel)
            # number_of_prod[0] += np.size(kernel)
            # number_of_memory_access[0] += 2 * np.size(kernel)

            # mode2
            # number_of_sum[1] += (np.size(kernel) - size_of_zero_weights)
            # number_of_prod[1] += (np.size(kernel) - size_of_zero_weights)

            # we read zero weights only for understanding that they are zero.
            # As soon as we understood, we ignore input data of that position
            # number_of_memory_access[1] += 2 * (np.size(kernel) - size_of_zero_weights) + size_of_zero_weights

            # mode3
            # number_of_sum[3] +=
            # -----------------------------------

            result[i, j] = temp_result
            j += stride
        i += stride
    return result, weights_inputs_common  # , number_of_sum, number_of_prod

cdef float multiply(np.ndarray input, tuple positions, float weight):
    cdef:
        int i = positions[0].shape[0]
        float sum = 0
    while i >= 0:
        i -= 1
        sum += input[positions[0][i], positions[1][i], positions[2][i]]
    return weight * sum

def get_common_regions(weights_inputs_common):
    common = {}
    for i in weights_inputs_common:
        for j in weights_inputs_common:
            if i != j and ((j, i) not in common.keys()):
                common[(i, j)] = intersect_along_first_axis(weights_inputs_common[i], weights_inputs_common[j])
    return common

def intersect_along_first_axis(a, b):
    # check that casting to void will create equal size elements
    assert a.shape[1:] == b.shape[1:]
    assert a.dtype == b.dtype

    # compute dtypes
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    orig_dt = np.dtype((a.dtype, a.shape[1:]))

    # convert to 1d void arrays
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    a_void = a.reshape(a.shape[0], -1).view(void_dt)
    b_void = b.reshape(b.shape[0], -1).view(void_dt)

    # intersect, then convert back
    return np.intersect1d(b_void, a_void).view(orig_dt)
