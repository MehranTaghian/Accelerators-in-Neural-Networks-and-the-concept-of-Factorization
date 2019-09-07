import tensorflow as tf
import numpy as np
cimport numpy as np
cimport cython
cimport cython
from zipfile import ZipFile
from zipfile import ZIP_DEFLATED
import csv
import os
import xlsxwriter as excel

cdef np.ndarray quantization(np.ndarray kernel, int number_of_weights = 16):
    cdef:
        np.npy_float32 r_max = np.max(kernel)
        np.npy_float32 r_min = np.min(kernel)
        np.ndarray quantized = np.int8(number_of_weights / (r_max - r_min) * kernel)
    return quantized

@cython.cdivision(True)
cpdef np.ndarray convolve(np.ndarray input_data, np.ndarray conv_layer, np.ndarray bias, str layer_num,
                          str padding="VALID", int stride=1, int number_of_weights = 16):
    """

    :param number_of_weights: for quantization, how many bits to dedicate to a weight
    :param layer_num: 
    :param input_data: input data to the kernel
    :param conv_layer: 4D convolution layer to be convolved into input_data
    :param bias:
    :param padding: if "SAME", we add padding to the input data so that
                    the output would be the same size as input
    :param stride:
    :return: result of convolution
    """
    print(input_data)
    cdef int p = 0
    cdef int i = 0
    cdef np.ndarray input_data_padded = input_data
    if padding == "SAME":
        p = int((conv_layer.shape[0] - 1) / 2)
        input_data_padded = np.zeros([input_data.shape[0] + 2 * p, input_data.shape[1] + 2 * p, input_data.shape[2]])
        while i < input_data.shape[2]:
            input_data_padded[:, :, i] = np.pad(input_data[:, :, i], (p, p), mode="constant")
            i += 1
    cdef int filter_num = conv_layer.shape[3]
    cdef int output_width = int((input_data_padded.shape[0] - conv_layer.shape[0]) / stride + 1)
    cdef list output_size = [output_width, output_width, filter_num]
    cdef np.ndarray result = np.zeros(output_size)
    cdef int number_of_sum, number_of_prod, number_of_memory_access
    cdef np.ndarray kernel, conv_result
    cdef np.ndarray weights, repeated
    cdef np.ndarray conv_layer_quantized = quantization(conv_layer)
    cdef dict common = {}
    cdef int j, k
    i = 0
    # number_of_common_inputs is for those inputs who have n common weights where 0 <= n <= number_of_weights
    cdef np.ndarray number_of_common_inputs = np.zeros(number_of_weights, dtype=int)

    workbook = excel.Workbook(f'result\\{layer_num}.xlsx')
    worksheet = workbook.add_worksheet(layer_num)
    unique_weights_in_kernel = np.unique(conv_layer)

    for f in range(filter_num):
        print("Filter", f)
        # number_of_zeros_without_quantization = np.where(conv_layer[:, :, :, f] == 0)[0].shape[0]
        # kernel = conv_layer_quantized[:, :, :, f]
        kernel = conv_layer[:, :, :, f]
        unique = {}
        for u in unique_weights_in_kernel:
            unique[u] = np.where(kernel == u)[0].shape[0]

        write_unique_to_excel(worksheet, layer_num, f, unique)

        # print(np.where(kernel == 0)[0].shape[0])
        # print(kernel.size)
        # while i < input_data_padded.shape[0]:
        #     j = 0
        #     while j < input_data_padded.shape[1]:
        #         k = 0
        #         while k < input_data_padded.shape[2]:
        #             common[(i, j, k)] = []
        #             k += 1
        #         j += 1
        #     i += 1

        # print('factoring')
        # weights, repeated = conv_factorization(kernel)
        # print('convolving')
        # conv_result, common, number_of_sum, number_of_prod, number_of_memory_access = conv2d(input_data_padded,
        #                                                                                      kernel,
        #                                                                                      repeated, weights,
        #                                                                                      common,
        #                                                                                      stride)
        # conv_result, common, number_of_sum, number_of_prod, number_of_memory_access = conv2d(input_data_padded,
        #                                                                                      kernel,
        #                                                                                      None, None,
        #                                                                                      None,
        #                                                                                      stride)

        # write_data_to_excel(worksheet, layer_num, f, number_of_memory_access, number_of_sum, number_of_prod)

        # print("Calculating Unique")
        # number_of_common_inputs = np.zeros(number_of_weights)
        # i = 0
        # while i < input_data_padded.shape[0]:
        #     j = 0
        #     while j < input_data_padded.shape[1]:
        #         k = 0
        #         while k < input_data_padded.shape[2]:
        #             number_of_common_inputs[len(np.unique(common[(i, j, k)]))] += 1
        #             k += 1
        #         j += 1
        #     i += 1
        #
        # result[:, :, f] = conv_result + bias[f]

        # Experimental part
        # -----------------------------
        # # common = get_common_regions(weights_inputs_common, weights)
        print("writing...")
        # write_data_to_excel(worksheet, layer_num, f, len(unique), np.where(kernel == 0)[0].shape[0], kernel.size)
        # write_number_of_uniques(number_of_common_inputs, layer_num, f)
        # write_common_regions_csv(common, 'layer-' + layer_num, f)

        # write_experiment_info_to_file('layer-' + layer_num, f, 'mode1', number_of_sum[0], number_of_prod[0],
        #                               number_of_memory_access[0])
        # write_experiment_info_to_file('layer-' + layer_num, f, 'mode2', number_of_sum[1], number_of_prod[1],
        #                               number_of_memory_access[1])
        # write_experiment_info_to_file('layer-' + layer_num, f, 'mode3', number_of_sum[2], number_of_prod[2],
        #                               number_of_memory_access[2])
        # write_experiment_info_to_file('layer-' + layer_num, f, 'mode4', number_of_sum[3], number_of_prod[3],
        #                               number_of_memory_access[3])

        # print("Maximum repeated weight:", max_index, "Number of repeated:", len(repeated_w[max_index][0]))
        # print("Minimum repeated weight: ", min_index, "Number of repeated:", len(repeated_w[min_index][0]))

        # print("Number of sum", number_of_sum)
        # print("Number of prod", number_of_prod)

        # -----------------------------------------

    workbook.close()
    return result

def write_unique_to_excel(worksheet, layer_num, filter_num, unique):
    if filter_num == 0:
        worksheet.write('A1', 'Filter')
        i = 66
        for k in unique:
            worksheet.write(f'{chr(i)}1', k)
            i += 1

    worksheet.write('A' + str(filter_num + 2), f'{layer_num}-Filter{filter_num}')
    i = 66
    for k in unique:
        worksheet.write(f'{chr(i)}{str(filter_num + 2)}', unique[k])
        i += 1

def write_data_to_excel(worksheet, layer_num, filter_num, memory, sum, prod):
    if filter_num == 0:
        worksheet.write('A1', 'Filter')
        worksheet.write('B1', '#memory_access')
        worksheet.write('C1', '#prod')
        worksheet.write('D1', '#sum')

    worksheet.write('A' + str(filter_num + 2), f'{layer_num}-Filter{filter_num}')
    worksheet.write('B' + str(filter_num + 2), memory)
    worksheet.write('C' + str(filter_num + 2), prod)
    worksheet.write('D' + str(filter_num + 2), sum)

cdef write_number_of_uniques(np.ndarray common_inputs, str layer, int filter_num):
    cdef int i = 0
    cdef int size = common_inputs.shape[0]

    if filter_num == 0:
        with open("result\\" + layer + ".txt", 'w') as f:
            f.write("Filter" + str(filter_num) + ":\n")
            while i < size:
                f.write(str(i) + ":" + str(common_inputs[i]) + '\n')
                i += 1
    else:
        with open("result\\" + layer + ".txt", 'a') as f:
            f.write("Filter" + str(filter_num) + ":\n")
            while i < size:
                f.write(str(i) + ":" + str(common_inputs[i]) + '\n')
                i += 1

cdef write_experiment_info_to_file(str layer, int filter_num, str mode, int number_of_sum, int number_of_prod,
                                   int number_of_memory_access):
    if filter_num == 0 and mode == 'mode1':
        with open("result\\" + layer + ".txt", 'w') as f:
            if mode == 'mode1':
                f.write("Filter" + str(filter_num) + ':\n')
            f.write(mode + "\n")
            f.write("Number of sum: " + str(number_of_sum) + '\n')
            f.write("Number of product: " + str(number_of_prod) + '\n')
            f.write("Number of memory access: " + str(number_of_memory_access) + '\n')
    else:
        with open("result\\" + layer + ".txt", 'a') as f:
            if mode == 'mode1':
                f.write("Filter" + str(filter_num) + ':\n')
            f.write(mode + "\n")
            f.write("Number of sum: " + str(number_of_sum) + '\n')
            f.write("Number of product: " + str(number_of_prod) + '\n')
            f.write("Number of memory access: " + str(number_of_memory_access) + '\n')

cdef write_common_regions_csv(dict common, str layer_name, int filter_num):
    csv.register_dialect('myDialect', delimiter=':', quoting=csv.QUOTE_ALL)
    cdef int i = 0, j = 0, k = 0
    cdef list keys = list(common.keys())
    cdef int size = len(keys)
    with open('result\\common-' + layer_name + "-Filter" + str(filter_num) + '.csv', 'w') as csvfile:
        # fieldnames = ['Position', 'weights']
        # writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect="myDialect")
        writer = csv.writer(csvfile)
        # writer.writeheader()
        writer.writerow("Filter" + str(filter_num))
        while i < size:
            writer.writerow([keys[i], common[keys[i]]])
            i += 1

    with ZipFile('VGG.zip', 'a', ZIP_DEFLATED) as zip:
        zip.write('result\\common-' + layer_name + "-Filter" + str(filter_num) + '.csv')
    os.remove('result\\common-' + layer_name + "-Filter" + str(filter_num) + '.csv')
    print("writing completed")

cdef conv_factorization(np.ndarray kernel):
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

        # Experimental part
        if len(temp[0]) > max:
            max = len(temp[0])
            max_index = i
        if len(temp[0]) < min:
            min = len(temp[0])
            min_index = i
        # ----------------------------------
        index = np.concatenate((temp[0][np.newaxis], temp[1][np.newaxis], temp[2][np.newaxis],), axis=0).T
        repeated = np.append(repeated, index, axis=0)
        indices[i] = len(index) + indices[i - 1] if i > 0 else len(index)
        i += 1

    indices = indices[:len(indices) - 1]
    return weight, np.array(np.split(repeated, indices), dtype=object) if len(indices) > 0 else repeated

@cython.cdivision(True)
cdef conv2d(np.ndarray data, np.ndarray kernel, np.ndarray repeated_position, np.ndarray weights, dict common,
            int stride=1):
    """
        The kernel is 3d like in image with 3 channels RGB
    :param weights: 
    :param data: is the actual input to convolution
    :param kernel: is the kernel of convolution e.g. a 3 by 3 kernel
    :param repeated_position: is the position of repeated weights
    :param stride: steps of moving kernel
    :return: the result of convolution
    """
    cdef:
        int result_size = int((data.shape[0] - kernel.shape[0]) / stride + 1)
        np.ndarray result = np.zeros([result_size, result_size])
        # np.ndarray number_of_sum = np.zeros(4, dtype=int)
        # np.ndarray number_of_prod = np.zeros(4, dtype=int)
        # np.ndarray number_of_memory_access = np.zeros(4, dtype=int)
        int size_of_zero_weights = 0
        int number_of_sum = 0
        int number_of_prod = 0
        int number_of_memory_access = 0
        # dict weights_inputs_common = {}
        # np.ndarray inputs_weights_common = np.zeros([data.shape[0], data.shape[1], data.shape[2], weights.shape[0]])

        # np.ndarray counter = np.zeros(weights.shape[0], dtype=int)
        np.ndarray counter = np.zeros(np.shape(data), dtype=int)

        float temp_result
        int width = ((data.shape[1] - kernel.shape[1]) / stride) + 1
        int height = ((data.shape[0] - kernel.shape[0]) / stride) + 1
        int i = 0, j = 0, ind = 0, key = 0
    #This loop is for result
    # while i < height:
    #     j = 0
    #     while j < width:
    #         temp_result = 0
    #         ind = 0
    #         while ind < weights.shape[0]:
    while i < (result.shape[0]):
        j = 0
        while j <= (result.shape[1]):
            ind = 0
            # number_of_memory_access += (2 * kernel.size - np.where(kernel == 0)[0].shape[0])
            # number_of_prod += kernel.size - np.where(kernel == 0)[0].shape[0]

            # number_of_memory_access += kernel.size * 2
            # number_of_prod += kernel.size
            # temp_result = 0
            # temp = np.where(kernel == 0)
            # temp2 = np.where(
            #     data[i * stride: i * stride + kernel.shape[0] - 1, j * stride: j * stride + kernel.shape[1] - 1,
            #     :] == 0)
            #
            # number_of_zero_weights = temp[0].shape[0]
            # number_of_zero_inputs = temp2[0].shape[0]
            #
            # zero_weights_pos = np.zeros([temp[0].shape[0], 3])
            # zero_inputs_pos = np.zeros([temp2[0].shape[0], 3])
            #
            # zero_weights_pos[:, 0] = temp[0]
            # zero_weights_pos[:, 1] = temp[1]
            # zero_weights_pos[:, 2] = temp[2]
            #
            # zero_inputs_pos[:, 0] = temp2[0]
            # zero_inputs_pos[:, 1] = temp2[1]
            # zero_inputs_pos[:, 2] = temp2[2]

            # zero_inputs_pos -= [i * stride, j * stride, 0]
            # print(f'zero input:{zero_inputs_pos}')
            # print(f'zero weights:{zero_weights_pos}')

            # number_of_both_zeros = 0
            # we = 0
            # inp = 0
            # while we < zero_weights_pos.shape[0]:
            #     inp = 0
            #     while inp < zero_inputs_pos.shape[0]:
            #         if (zero_weights_pos[we, :] == zero_inputs_pos[inp, :]).all():
            #             number_of_both_zeros += 1
            #         inp += 1
            #     we += 1
            # if zero_inputs_pos.shape[0] > 0 and zero_weights_pos.shape[0] > 0:
            #     number_of_both_zeros = len(intersect_along_first_axis(zero_weights_pos, zero_inputs_pos))

            # number_of_sum += kernel.size - (number_of_zero_inputs + number_of_zero_weights - number_of_both_zeros)
            # number_of_prod += kernel.size - (number_of_zero_inputs + number_of_zero_weights - number_of_both_zeros)
            # number_of_memory_access += 2 * (
            #         kernel.size - (number_of_zero_inputs + number_of_zero_weights - number_of_both_zeros))

            while ind < weights.shape[0]:
                if weights[ind] != 0:
                    number_of_memory_access += 1 + repeated_position[ind].shape[0]
                    number_of_sum += repeated_position[ind].shape[0]
                    if repeated_position[ind].shape[0]:
                        number_of_sum += 1
                    #count number of zero inputs:
                    # print(data[int(repeated_position[ind][:, 0]), int(repeated_position[ind][:, 1]), int(repeated_position[ind][:, 2])])
                    # indexes = np.array(repeated_position[ind], dtype=int)
                    # p = 0
                    #
                    # while p < indexes.shape[0]:
                    #     if indexes[p, 0] + i * stride >= data.shape[0] or indexes[p, 1] + j * stride >= data.shape[1]:
                    #         indexes = np.delete(indexes, p, axis=0)
                    #     else:
                    #         p += 1
                    #
                    # number_of_zero_inputs = np.where(
                    #     data[indexes[:, 0] + i * stride, indexes[:, 1] + j * stride, indexes[:, 2]] == 0)[0].shape[0]
                    # number_of_memory_access -= number_of_zero_inputs
                    # number_of_sum += repeated_position[ind].shape[0] - number_of_zero_inputs
                    # number_of_sum += repeated_position[ind].shape[0]
                    # number_of_sum += 1  #sum weight with other weights in global
                # if weights[ind] != 0:
                # number_of_prod += 1
                # number_of_sum += repeated_position[ind].shape[0]
                #     # mode4
                #     input_size = repeated_position[ind].shape[0]
                #     number_of_sum[3] += input_size
                #     number_of_prod[3] += 1
                #     number_of_memory_access[3] += input_size + 1
                #     # -----------------------------------
                # else:
                #     size_of_zero_weights = repeated_position[ind].shape[0]

                # Convolution inside the kernel
                # if weights[ind] != 0:  # Zero weights
                #     temp_result += multiply(data, repeated_position[ind], weights[ind], i, j)

                #---------------------------------------------------------------------
                #Expeimental part 3 for common as parameter---------------------------
                # key = 0
                # if weights.shape[0] > 1:
                #     keys = list(map(tuple, repeated_position[ind] + [i, j, 0]))
                #     while key < len(keys):
                #         common[keys[key]] += [weights[ind]]
                #         key += 1
                # else:
                #     keys = list(map(tuple, repeated_position + [i, j, 0]))
                #     while key < len(keys):
                #         common[keys[key]] += [weights[ind]]
                #         key += 1
                #---------------------------------------------------------------------

                # np.append(weights_inputs_common[ind],
                #           np.concatenate(
                #               ((repeated_position[ind][:, 0] + i)[np.newaxis],
                #                (repeated_position[ind][:, 1] + j)[np.newaxis],
                #                (repeated_position[ind][:, 2])[np.newaxis]),
                #               axis=0).T, axis=0)
                ind += 1
                number_of_prod += 1

            # # mode1
            # number_of_sum[0] += np.size(kernel)
            # number_of_prod[0] += np.size(kernel)
            # number_of_memory_access[0] += 2 * np.size(kernel)
            #
            # # mode3
            # number_of_sum[2] += (np.size(kernel) - size_of_zero_weights)
            # number_of_prod[2] += (np.size(kernel) - size_of_zero_weights)
            # number_of_memory_access[2] += 2 * (np.size(kernel) - size_of_zero_weights)

            # result[i, j] = temp_result
            j += 1
        i += 1

        #this part for i and j are for result
        #     j += 1
        # i += 1
    return result, common, number_of_sum, number_of_prod, number_of_memory_access  # , number_of_sum, number_of_prod

cdef float multiply(np.ndarray input, np.ndarray positions, float weight, int i, int j):
    cdef:
        int k = positions.shape[0]
        float sum = 0
    while k > 0:
        k -= 1
        sum += input[positions[k][0] + i, positions[k][1] + j, positions[k][2]]
    return weight * sum

cdef get_common_regions(weights_inputs_common, weights):
    cdef:
        dict common = {}
        int i = 0, j = 0
        int size = len(weights)

    while i < size:
        j = 0
        while j < size:
            if i != j and ((weights[j], weights[i]) not in common.keys()):
                common[(weights[i], weights[j])] = intersect_along_first_axis(weights_inputs_common[i],
                                                                              weights_inputs_common[j])
            j += 1
        i += 1
    return common

cpdef np.ndarray intersect_along_first_axis(a, b):
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

# if weights.shape[0] > 1:
#     inputs_weights_common[
#         np.array(repeated_position[ind][:, 0] + i, dtype=int),
#         np.array(repeated_position[ind][:, 1] + j, dtype=int),
#         np.array(repeated_position[ind][:, 2], dtype=int), ind] = weights[ind]
# else:
#     inputs_weights_common[
#         repeated_position[:, 0] + i, repeated_position[:, 1] + j, repeated_position[:,
#                                                                   2], ind] = weights[ind]
# Experimental part 1 for weights inputs common ---------------------
#     if i == j == 0:
#         weights_inputs_common[ind] = np.zeros([width * height * repeated_position[ind].shape[0], 3])
#
#     weights_inputs_common[ind][
#     counter[ind] * repeated_position[ind].shape[0]: (counter[ind] + 1) * repeated_position[ind].shape[0]] = \
#         (repeated_position[ind] + [i, j, 0])
#     # weights_inputs_common[ind] += (repeated_position[ind] + [i, j, 0]).tolist()
#     counter[ind] += 1
#--------------------------------------------------------------------
