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


class Experiment:
    def __init__(self):
        self.exp_quantized_without_zero_weights = ExperimentalData()
        self.exp_quantized_without_zero_weights_and_inputs = ExperimentalData()
        self.exp_quantized_factorized_without_zero_weights = ExperimentalData()
        self.exp_quantized_factorized_without_zero_weights_and_inputs = ExperimentalData()
        self.exp_factorized_without_zero_weights = ExperimentalData()
        self.exp_factorized_without_zero_weights_and_inputs = ExperimentalData()
        self.exp_base = ExperimentalData()
        self.exp_base_x_0 = ExperimentalData()
        self.exp_base_w_0 = ExperimentalData()
        self.exp_base_w_x_0 = ExperimentalData()


@cython.cdivision(True)
cpdef np.ndarray convolve(general_worksheet, worksheet, np.ndarray input_data, np.ndarray conv_layer, np.ndarray bias,
                          experiment,
                          int layer_num,
                          str mode,
                          str padding="VALID", int stride=1, int number_of_weights = 16):
    """
    :param worksheet: worksheet for writing result into excel file
    :param mode: mode is either QUANTIZED or NON_QUANTIZED
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
    # cdef np.ndarray conv_layer_quantized = quantization(conv_layer)
    cdef dict common = {}
    cdef int j, k
    i = 0
    # number_of_common_inputs is for those inputs who have n common weights where 0 <= n <= number_of_weights
    cdef np.ndarray number_of_common_inputs = np.zeros(number_of_weights, dtype=int)

    for f in range(filter_num):

        worksheet.write('A' + str(f + 3), f'Filter{f}')

        print("Filter", f)
        kernel = conv_layer[:, :, :, f]

        if mode == "QUANTIZED":

            #   Without factorization part (Quantized only)
            exp_quantized_without_zero_weights, exp_quantized_without_zero_weights_and_inputs = \
                conv2d(input_data_padded, kernel, 'NON_FACTORIZED', True, None, None, stride)

            experiment.exp_quantized_without_zero_weights.number_of_memory_access += exp_quantized_without_zero_weights.number_of_memory_access
            experiment.exp_quantized_without_zero_weights.number_of_add += exp_quantized_without_zero_weights.number_of_add
            experiment.exp_quantized_without_zero_weights.number_of_multiply += exp_quantized_without_zero_weights.number_of_multiply

            experiment.exp_quantized_without_zero_weights_and_inputs.number_of_memory_access += exp_quantized_without_zero_weights_and_inputs.number_of_memory_access
            experiment.exp_quantized_without_zero_weights_and_inputs.number_of_add += exp_quantized_without_zero_weights_and_inputs.number_of_add
            experiment.exp_quantized_without_zero_weights_and_inputs.number_of_multiply += exp_quantized_without_zero_weights_and_inputs.number_of_multiply

            write_data_to_excel(worksheet, layer_num, f, 'quantized', exp_quantized_without_zero_weights,
                                exp_quantized_without_zero_weights_and_inputs)

            #   With factorization part (Factorized-Quantized)
            weights, repeated = conv_factorization(kernel)
            exp_quantized_factorized_without_zero_weights, exp_quantized_factorized_without_zero_weights_and_inputs = \
                conv2d(input_data_padded, kernel, 'FACTORIZED', True, repeated, weights, stride)

            experiment.exp_quantized_factorized_without_zero_weights.number_of_memory_access += exp_quantized_factorized_without_zero_weights.number_of_memory_access
            experiment.exp_quantized_factorized_without_zero_weights.number_of_add += exp_quantized_factorized_without_zero_weights.number_of_add
            experiment.exp_quantized_factorized_without_zero_weights.number_of_multiply += exp_quantized_factorized_without_zero_weights.number_of_multiply

            experiment.exp_quantized_factorized_without_zero_weights_and_inputs.number_of_memory_access += exp_quantized_factorized_without_zero_weights_and_inputs.number_of_memory_access
            experiment.exp_quantized_factorized_without_zero_weights_and_inputs.number_of_add += exp_quantized_factorized_without_zero_weights_and_inputs.number_of_add
            experiment.exp_quantized_factorized_without_zero_weights_and_inputs.number_of_multiply += exp_quantized_factorized_without_zero_weights_and_inputs.number_of_multiply

            write_data_to_excel(worksheet, layer_num, f, 'quantized+factorized',
                                exp_quantized_factorized_without_zero_weights,
                                exp_quantized_factorized_without_zero_weights_and_inputs)

        elif mode == "NON_QUANTIZED":

            #   Without factorization part (Base mode)
            exp_base, exp_base_x_0, exp_base_w_0, exp_base_w_x_0 = \
                conv2d(input_data_padded, kernel, 'NON_FACTORIZED', False, None, None, stride)
            write_data_to_excel(worksheet, layer_num, f, 'base',
                                exp_base, exp_base_w_0, exp_base_x_0, exp_base_w_x_0)

            experiment.exp_base.number_of_memory_access += exp_base.number_of_memory_access
            experiment.exp_base.number_of_add += exp_base.number_of_add
            experiment.exp_base.number_of_multiply += exp_base.number_of_multiply

            experiment.exp_base_x_0.number_of_memory_access += exp_base_x_0.number_of_memory_access
            experiment.exp_base_x_0.number_of_add += exp_base_x_0.number_of_add
            experiment.exp_base_x_0.number_of_multiply += exp_base_x_0.number_of_multiply

            experiment.exp_base_w_0.number_of_memory_access += exp_base_w_0.number_of_memory_access
            experiment.exp_base_w_0.number_of_add += exp_base_w_0.number_of_add
            experiment.exp_base_w_0.number_of_multiply += exp_base_w_0.number_of_multiply

            experiment.exp_base_w_x_0.number_of_memory_access += exp_base_w_x_0.number_of_memory_access
            experiment.exp_base_w_x_0.number_of_add += exp_base_w_x_0.number_of_add
            experiment.exp_base_w_x_0.number_of_multiply += exp_base_w_x_0.number_of_multiply

            #   With factorization part (Factorization without Quantization)
            weights, repeated = conv_factorization(kernel)
            exp_factorized_without_zero_weights, exp_factorized_without_zero_weights_and_inputs = \
                conv2d(input_data_padded, kernel, 'FACTORIZED', False, repeated, weights, stride)
            write_data_to_excel(worksheet, layer_num, f, 'factorized',
                                exp_factorized_without_zero_weights, exp_factorized_without_zero_weights_and_inputs)

            experiment.exp_factorized_without_zero_weights.number_of_memory_access += exp_factorized_without_zero_weights.number_of_memory_access
            experiment.exp_factorized_without_zero_weights.number_of_add += exp_factorized_without_zero_weights.number_of_add
            experiment.exp_factorized_without_zero_weights.number_of_multiply += exp_factorized_without_zero_weights.number_of_multiply

            experiment.exp_factorized_without_zero_weights_and_inputs.number_of_memory_access += exp_factorized_without_zero_weights_and_inputs.number_of_memory_access
            experiment.exp_factorized_without_zero_weights_and_inputs.number_of_add += exp_factorized_without_zero_weights_and_inputs.number_of_add
            experiment.exp_factorized_without_zero_weights_and_inputs.number_of_multiply += exp_factorized_without_zero_weights_and_inputs.number_of_multiply

    write_overal_data_to_excel(general_worksheet, experiment, layer_num)
    return result

def write_overal_data_to_excel(worksheet, experiment, layer):
    if layer == 1:
        worksheet.write('A1', 'Layer')
        worksheet.write('B2', 'base')
        worksheet.write('C2', 'base-w-0')
        worksheet.write('D2', 'base-x-0')
        worksheet.write('E2', 'base-w-x-0')
        worksheet.write('F2', 'quantized (Without zero weights)')
        worksheet.write('G2', 'quantized (Without zero weights and inputs)')
        worksheet.write('H2', 'factorized (Without zero weights)')
        worksheet.write('I2', 'factorized (Without zero weights and inputs)')
        worksheet.write('J2', 'factorized+quantized (Without zero weights)')
        worksheet.write('K2', 'factorized+quantized (Without zero weights and inputs)')
        worksheet.write('M2', 'base')
        worksheet.write('N2', 'base-w-0')
        worksheet.write('O2', 'base-x-0')
        worksheet.write('P2', 'base-w-x-0')
        worksheet.write('Q2', 'quantized (Without zero weights)')
        worksheet.write('R2', 'quantized (Without zero weights and inputs)')
        worksheet.write('S2', 'factorized (Without zero weights)')
        worksheet.write('T2', 'factorized (Without zero weights and inputs)')
        worksheet.write('U2', 'factorized+quantized (Without zero weights)')
        worksheet.write('V2', 'factorized+quantized (Without zero weights and inputs)')
        worksheet.write('X2', 'base')
        worksheet.write('Y2', 'base-w-0')
        worksheet.write('Z2', 'base-x-0')
        worksheet.write('AA2', 'base-w-x-0')
        worksheet.write('AB2', 'quantized (Without zero weights)')
        worksheet.write('AC2', 'quantized (Without zero weights and inputs)')
        worksheet.write('AD2', 'factorized (Without zero weights)')
        worksheet.write('AE2', 'factorized (Without zero weights and inputs)')
        worksheet.write('AF2', 'factorized+quantized (Without zero weights)')
        worksheet.write('AG2', 'factorized+quantized (Without zero weights and inputs)')

    worksheet.write(f'A{layer + 2}', f'layer_{layer}')
    worksheet.write(f'B{layer + 2}', experiment.exp_base.number_of_memory_access)
    worksheet.write(f'C{layer + 2}', experiment.exp_base_w_0.number_of_memory_access)
    worksheet.write(f'D{layer + 2}', experiment.exp_base_x_0.number_of_memory_access)
    worksheet.write(f'E{layer + 2}', experiment.exp_base_w_x_0.number_of_memory_access)
    worksheet.write(f'F{layer + 2}', experiment.exp_quantized_without_zero_weights.number_of_memory_access)
    worksheet.write(f'G{layer + 2}', experiment.exp_quantized_without_zero_weights_and_inputs.number_of_memory_access)
    worksheet.write(f'H{layer + 2}', experiment.exp_factorized_without_zero_weights.number_of_memory_access)
    worksheet.write(f'I{layer + 2}', experiment.exp_factorized_without_zero_weights_and_inputs.number_of_memory_access)
    worksheet.write(f'J{layer + 2}', experiment.exp_quantized_factorized_without_zero_weights.number_of_memory_access)
    worksheet.write(f'K{layer + 2}',
                    experiment.exp_quantized_factorized_without_zero_weights_and_inputs.number_of_memory_access)

    worksheet.write(f'M{layer + 2}', experiment.exp_base.number_of_multiply)
    worksheet.write(f'N{layer + 2}', experiment.exp_base_w_0.number_of_multiply)
    worksheet.write(f'O{layer + 2}', experiment.exp_base_x_0.number_of_multiply)
    worksheet.write(f'P{layer + 2}', experiment.exp_base_w_x_0.number_of_multiply)
    worksheet.write(f'Q{layer + 2}', experiment.exp_quantized_without_zero_weights.number_of_multiply)
    worksheet.write(f'R{layer + 2}', experiment.exp_quantized_without_zero_weights_and_inputs.number_of_multiply)
    worksheet.write(f'S{layer + 2}', experiment.exp_factorized_without_zero_weights.number_of_multiply)
    worksheet.write(f'T{layer + 2}', experiment.exp_factorized_without_zero_weights_and_inputs.number_of_multiply)
    worksheet.write(f'U{layer + 2}', experiment.exp_quantized_factorized_without_zero_weights.number_of_multiply)
    worksheet.write(f'V{layer + 2}',
                    experiment.exp_quantized_factorized_without_zero_weights_and_inputs.number_of_multiply)

    worksheet.write(f'X{layer + 2}', experiment.exp_base.number_of_add)
    worksheet.write(f'Y{layer + 2}', experiment.exp_base_w_0.number_of_add)
    worksheet.write(f'Z{layer + 2}', experiment.exp_base_x_0.number_of_add)
    worksheet.write(f'AA{layer + 2}', experiment.exp_base_w_x_0.number_of_add)
    worksheet.write(f'AB{layer + 2}', experiment.exp_quantized_without_zero_weights.number_of_add)
    worksheet.write(f'AC{layer + 2}', experiment.exp_quantized_without_zero_weights_and_inputs.number_of_add)
    worksheet.write(f'AD{layer + 2}', experiment.exp_factorized_without_zero_weights.number_of_add)
    worksheet.write(f'AE{layer + 2}', experiment.exp_factorized_without_zero_weights_and_inputs.number_of_add)
    worksheet.write(f'AF{layer + 2}', experiment.exp_quantized_factorized_without_zero_weights.number_of_add)
    worksheet.write(f'AG{layer + 2}', experiment.exp_quantized_factorized_without_zero_weights_and_inputs.number_of_add)

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

def write_data_to_excel(worksheet, layer_num, filter_num, mode, *args):
    if filter_num == 0:
        worksheet.write('A1', 'Filter')
        worksheet.write('B2', 'base')
        worksheet.write('C2', 'base-w-0')
        worksheet.write('D2', 'base-x-0')
        worksheet.write('E2', 'base-w-x-0')
        worksheet.write('F2', 'quantized (Without zero weights)')
        worksheet.write('G2', 'quantized (Without zero weights and inputs)')
        worksheet.write('H2', 'factorized (Without zero weights)')
        worksheet.write('I2', 'factorized (Without zero weights and inputs)')
        worksheet.write('J2', 'factorized+quantized (Without zero weights)')
        worksheet.write('K2', 'factorized+quantized (Without zero weights and inputs)')

        worksheet.write('M2', 'base')
        worksheet.write('N2', 'base-w-0')
        worksheet.write('O2', 'base-x-0')
        worksheet.write('P2', 'base-w-x-0')
        worksheet.write('Q2', 'quantized (Without zero weights)')
        worksheet.write('R2', 'quantized (Without zero weights and inputs)')
        worksheet.write('S2', 'factorized (Without zero weights)')
        worksheet.write('T2', 'factorized (Without zero weights and inputs)')
        worksheet.write('U2', 'factorized+quantized (Without zero weights)')
        worksheet.write('V2', 'factorized+quantized (Without zero weights and inputs)')

        worksheet.write('X2', 'base')
        worksheet.write('Y2', 'base-w-0')
        worksheet.write('Z2', 'base-x-0')
        worksheet.write('AA2', 'base-w-x-0')
        worksheet.write('AB2', 'quantized (Without zero weights)')
        worksheet.write('AC2', 'quantized (Without zero weights and inputs)')
        worksheet.write('AD2', 'factorized (Without zero weights)')
        worksheet.write('AE2', 'factorized (Without zero weights and inputs)')
        worksheet.write('AF2', 'factorized+quantized (Without zero weights)')
        worksheet.write('AG2', 'factorized+quantized (Without zero weights and inputs)')

    if mode == 'base':
        worksheet.write('B' + str(filter_num + 3), args[0].number_of_memory_access)
        worksheet.write('C' + str(filter_num + 3), args[1].number_of_memory_access)
        worksheet.write('D' + str(filter_num + 3), args[2].number_of_memory_access)
        worksheet.write('E' + str(filter_num + 3), args[3].number_of_memory_access)

        worksheet.write('M' + str(filter_num + 3), args[0].number_of_multiply)
        worksheet.write('N' + str(filter_num + 3), args[1].number_of_multiply)
        worksheet.write('O' + str(filter_num + 3), args[2].number_of_multiply)
        worksheet.write('P' + str(filter_num + 3), args[3].number_of_multiply)

        worksheet.write('X' + str(filter_num + 3), args[0].number_of_add)
        worksheet.write('Y' + str(filter_num + 3), args[1].number_of_add)
        worksheet.write('Z' + str(filter_num + 3), args[2].number_of_add)
        worksheet.write('AA' + str(filter_num + 3), args[3].number_of_add)

    elif mode == 'quantized':
        worksheet.write('F' + str(filter_num + 3), args[0].number_of_memory_access)
        worksheet.write('G' + str(filter_num + 3), args[1].number_of_memory_access)

        worksheet.write('Q' + str(filter_num + 3), args[0].number_of_multiply)
        worksheet.write('R' + str(filter_num + 3), args[1].number_of_multiply)

        worksheet.write('AB' + str(filter_num + 3), args[0].number_of_add)
        worksheet.write('AC' + str(filter_num + 3), args[1].number_of_add)

    elif mode == 'factorized':
        worksheet.write('H' + str(filter_num + 3), args[0].number_of_memory_access)
        worksheet.write('I' + str(filter_num + 3), args[1].number_of_memory_access)

        worksheet.write('S' + str(filter_num + 3), args[0].number_of_multiply)
        worksheet.write('T' + str(filter_num + 3), args[1].number_of_multiply)

        worksheet.write('AD' + str(filter_num + 3), args[0].number_of_add)
        worksheet.write('AE' + str(filter_num + 3), args[1].number_of_add)

    elif mode == 'quantized+factorized':
        worksheet.write('J' + str(filter_num + 3), args[0].number_of_memory_access)
        worksheet.write('K' + str(filter_num + 3), args[1].number_of_memory_access)

        worksheet.write('U' + str(filter_num + 3), args[0].number_of_multiply)
        worksheet.write('V' + str(filter_num + 3), args[1].number_of_multiply)

        worksheet.write('AF' + str(filter_num + 3), args[0].number_of_add)
        worksheet.write('AG' + str(filter_num + 3), args[1].number_of_add)

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
        index = np.concatenate((temp[0][np.newaxis], temp[1][np.newaxis], temp[2][np.newaxis],), axis=0).T
        repeated = np.append(repeated, index, axis=0)
        indices[i] = len(index) + indices[i - 1] if i > 0 else len(index)
        i += 1

    indices = indices[:len(indices) - 1]
    return weight, np.array(np.split(repeated, indices), dtype=object) if len(indices) > 0 else repeated

@cython.cdivision(True)
cdef conv2d(np.ndarray data, np.ndarray kernel, mode, is_quantized, np.ndarray repeated_position, np.ndarray weights,
            int stride=1):
    """
        The kernel is 3d like in image with 3 channels RGB
    :param is_quantized: Whether the weights of network are quantized or not
    :param mode: It is whether FACTORIZED or NON_FACTORIZED
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
        int i = 0, j = 0, ind = 0, key = 0

    if mode == 'NON_FACTORIZED' and (not is_quantized):
        exp_base = ExperimentalData()
        exp_base_w_0 = ExperimentalData()
        exp_base_x_0 = ExperimentalData()
        exp_base_w_x_0 = ExperimentalData()

    elif mode == 'FACTORIZED' and is_quantized:
        exp_without_zero_weights = ExperimentalData()
        exp_without_zero_weights_and_inputs = ExperimentalData()

    elif mode == 'FACTORIZED' and (not is_quantized):
        exp_without_zero_weights = ExperimentalData()
        exp_without_zero_weights_and_inputs = ExperimentalData()

    elif mode == 'NON_FACTORIZED' and is_quantized:
        exp_quantized_without_zero_weights = ExperimentalData()
        exp_quantized_without_zero_weights_and_inputs = ExperimentalData()

    while i < (result.shape[0]):
        j = 0
        while j < (result.shape[1]):
            if mode == 'FACTORIZED':
                factorized_without_zero_weights(exp_without_zero_weights, weights, repeated_position)
                factorized_without_zero_weights_and_inputs(exp_without_zero_weights_and_inputs, data, weights,
                                                           repeated_position, stride, i, j)
            else:
                if is_quantized:
                    not_factorized_without_zero_weights(exp_quantized_without_zero_weights, kernel)
                    not_factorization_without_zero_weights_and_inputs(exp_quantized_without_zero_weights_and_inputs,
                                                                      data, kernel, stride, i, j)

                else:  #Base mode
                    base(exp_base, kernel)
                    base_x_0(exp_base_x_0, data, kernel, stride, i, j)
                    not_factorized_without_zero_weights(exp_base_w_0, kernel)
                    not_factorization_without_zero_weights_and_inputs(exp_base_w_x_0, data, kernel, stride, i, j)

            j += 1
        i += 1

    if mode == 'FACTORIZED':
        return exp_without_zero_weights, exp_without_zero_weights_and_inputs
    else:
        if is_quantized:
            return exp_quantized_without_zero_weights, exp_quantized_without_zero_weights_and_inputs
        else:  #Base mode
            return exp_base, exp_base_x_0, exp_base_w_0, exp_base_w_x_0


class ExperimentalData:

    def __init__(self):
        self.number_of_memory_access = 0
        self.number_of_multiply = 0
        self.number_of_add = 0


def base(exp: ExperimentalData, kernel):
    exp.number_of_memory_access += 2 * kernel.size
    exp.number_of_add += kernel.size
    exp.number_of_multiply += kernel.size

def base_x_0(exp, data, kernel, stride, i, j):
    """
    This function is for calculating number of sum, memory access and product
    for base method without any factorization and quantization, by only neglecting
    zero inputs
    :param exp: Experimental Data
    :param data: input data
    :param kernel:
    :param stride:
    :param i: row index of kernel over input
    :param j: col index of kernel over input
    :return:
    """
    number_of_zero_inputs = np.where(
        data[i * stride: i * stride + kernel.shape[0] - 1, j * stride: j * stride + kernel.shape[1] - 1, :] == 0)[
        0].shape[0]
    exp.number_of_memory_access += 2 * (kernel.size - number_of_zero_inputs)
    exp.number_of_add += kernel.size - number_of_zero_inputs
    exp.number_of_multiply += kernel.size - number_of_zero_inputs

def not_factorized_without_zero_weights(exp, kernel):
    """
    This function is for calculating the number of sum, memory access and product for
    both methods of quantized and not quantized
    :param exp: ExperimentalData
    :param kernel:
    """
    number_of_zero_weights = np.where(kernel == 0)[0].shape[0]
    exp.number_of_memory_access += 2 * (kernel.size - number_of_zero_weights)
    exp.number_of_add += kernel.size - number_of_zero_weights
    exp.number_of_multiply += kernel.size - number_of_zero_weights

def not_factorization_without_zero_weights_and_inputs(exp, data, kernel, stride, i, j):
    """
    This function is for calculating the number of sum, memory access and product for
    both methods of quantized and not quantized
    :param exp: ExperimentalData
    :param data: input data
    :param kernel:
    :param stride:
    :param i: row index of kernel over input
    :param j: col index of kernel over input
    """
    temp = np.where(kernel == 0)
    temp2 = np.where(
        data[i * stride: i * stride + kernel.shape[0], j * stride: j * stride + kernel.shape[1],
        :] == 0)

    number_of_zero_weights = temp[0].shape[0]
    number_of_zero_inputs = temp2[0].shape[0]

    zero_weights_pos = np.zeros([temp[0].shape[0], 3])
    zero_inputs_pos = np.zeros([temp2[0].shape[0], 3])

    zero_weights_pos[:, 0] = temp[0]
    zero_weights_pos[:, 1] = temp[1]
    zero_weights_pos[:, 2] = temp[2]

    zero_inputs_pos[:, 0] = temp2[0]
    zero_inputs_pos[:, 1] = temp2[1]
    zero_inputs_pos[:, 2] = temp2[2]

    number_of_both_zeros = 0

    if zero_inputs_pos.shape[0] > 0 and zero_weights_pos.shape[0] > 0:
        number_of_both_zeros = len(intersect_along_first_axis(zero_weights_pos, zero_inputs_pos))

    exp.number_of_add += kernel.size - (number_of_zero_inputs + number_of_zero_weights - number_of_both_zeros)
    exp.number_of_multiply += kernel.size - (number_of_zero_inputs + number_of_zero_weights - number_of_both_zeros)
    exp.number_of_memory_access += 2 * (kernel.size - (
            number_of_zero_inputs + number_of_zero_weights - number_of_both_zeros))

def factorized_without_zero_weights(exp, unique_weights, positions):
    """
    This function is for calculating the number of sum, memory access and product for
    both method of factorized, with and without quantized
    :param exp: Experimental Data
    :param unique_weights: set of factored weights
    :param positions: repeated positions inside input data corresponded to the factored weights
    """
    ind = 0
    while ind < unique_weights.shape[0]:
        if unique_weights[ind] != 0:
            exp.number_of_memory_access += positions[ind].shape[0] + 1
            exp.number_of_add += positions[ind].shape[0]
            exp.number_of_multiply += 1
        ind += 1

def factorized_without_zero_weights_and_inputs(exp, data, unique_weights, positions, stride, i, j):
    """
    This function is for calculating the number of sum, memory access and product for
    both method of factorized, with and without quantized
    :param exp: Experimental Data
    :param data: input data
    :param unique_weights: set of factored weights
    :param positions: repeated positions inside input data corresponded to the factored weights
    :param stride:
    :param i: row index of kernel over input data
    :param j: col index of kernel over input data
    """
    ind = 0
    while ind < unique_weights.shape[0]:
        if unique_weights[ind] != 0:
            exp.number_of_memory_access += positions[ind].shape[0] + 1
            exp.number_of_add += positions[ind].shape[0]

            temp_indices = np.array(positions[ind], dtype=int)
            p = 0
            indexes = np.empty([0, 3], dtype=int)

            while p < temp_indices.shape[0]:
                if temp_indices[p, 0] + i * stride < data.shape[0] and temp_indices[p, 1] + j * stride < data.shape[1]:
                    # temp_indices = np.delete(temp_indices, p, axis=0)
                    indexes = np.append(indexes, temp_indices[p, :][np.newaxis], axis=0)
                p += 1

            number_of_zero_inputs = np.where(
                data[indexes[:, 0] + i * stride, indexes[:, 1] + j * stride, indexes[:, 2]] == 0)[0].shape[0]

            exp.number_of_memory_access -= number_of_zero_inputs
            exp.number_of_add -= number_of_zero_inputs
            if number_of_zero_inputs < positions[ind].shape[0]:
                exp.number_of_multiply += 1
            else:  # we do not read weights that are corresponded to zero inputs
                exp.number_of_memory_access -= 1

        ind += 1

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
