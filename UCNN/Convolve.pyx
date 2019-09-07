import cython
import numpy as np
cimport numpy as np
from UCNN_lane import conv_single_stride
import xlsxwriter as excel

@cython.cdivision(True)
cpdef conv2d(np.ndarray data, np.ndarray filter1, np.ndarray filter2, int stride=1):
    """
        The kernel is 3d like in image with 3 channels RGB
    :param filter1:
    :param filter2:
    :param data: is the actual input to convolution
    :param stride: steps of moving kernel
    :return: the result of convolution
    """
    cdef:
        int result_size = int((data.shape[0] - filter1.shape[0]) / stride + 1)
        np.ndarray result = np.zeros([result_size, result_size, 2])
        int width = ((data.shape[1] - filter1.shape[1]) / stride) + 1
        int height = ((data.shape[0] - filter1.shape[0]) / stride) + 1
        int i = 0, j = 0

        int number_of_memory_access_weights = 0, number_of_memory_access_input = 0, \
            number_of_access_to_accumulator2 = 0, number_of_access_to_accumulator3 = 0, \
            number_of_access_to_multiplier = 0, wiT1_table_size = 0, wiT2_table_size = 0, input_table_size = 0, \
            access_input_buffer = 0, access_weight_buffer = 0, access_partial_sum_buffer = 0

    while i < (result.shape[0]):
        j = 0
        while j < (result.shape[1]):
            # print(data[i: i + filter1.shape[0], j: j + filter1.shape[1], :])
            row_index, col_index = i * stride, j * stride
            result[i, j, :], temp_number_of_memory_access_weights, temp_number_of_memory_access_input, \
            temp_number_of_access_to_accumulator2, temp_number_of_access_to_accumulator3, \
            temp_number_of_access_to_multiplier, temp_wiT1_table_size, temp_wiT2_table_size, temp_input_table_size, \
            temp_access_input_buffer, temp_access_weight_buffer, temp_access_partial_sum_buffer = conv_single_stride(
                filter1, filter2,
                data[row_index: row_index + filter1.shape[0],
                col_index: col_index + filter1.shape[1], :])

            number_of_memory_access_weights += temp_number_of_memory_access_weights
            number_of_memory_access_input += temp_number_of_memory_access_input
            number_of_access_to_multiplier += temp_number_of_access_to_multiplier
            number_of_access_to_accumulator3 += temp_number_of_access_to_accumulator3
            number_of_access_to_accumulator2 += temp_number_of_access_to_accumulator2
            wiT1_table_size = temp_wiT1_table_size if temp_wiT1_table_size > wiT1_table_size else wiT1_table_size
            wiT2_table_size = temp_wiT2_table_size if temp_wiT2_table_size > wiT2_table_size else wiT2_table_size
            input_table_size = temp_input_table_size if temp_input_table_size > input_table_size else input_table_size
            access_partial_sum_buffer += temp_access_partial_sum_buffer
            access_weight_buffer += temp_access_weight_buffer
            access_input_buffer += temp_access_weight_buffer

            j += 1
        i += 1

    return result, number_of_memory_access_weights, number_of_memory_access_input, \
           number_of_access_to_accumulator2, number_of_access_to_accumulator3, \
           number_of_access_to_multiplier, wiT1_table_size, wiT2_table_size, input_table_size, \
           access_input_buffer, access_weight_buffer, access_partial_sum_buffer

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
    workbook = excel.Workbook(f'result\\{layer_num}.xlsx')
    worksheet = workbook.add_worksheet(layer_num)
    for i in range(0, filter_num, 2):
        result[:, :, i: i + 2], number_of_memory_access_weights, number_of_memory_access_input, \
        number_of_access_to_accumulator2, number_of_access_to_accumulator3, \
        number_of_access_to_multiplier, wiT1_table_size, wiT2_table_size, input_table_size, \
        access_input_buffer, access_weight_buffer, access_partial_sum_buffer = conv2d(input_data_padded,
                                                                                      kernel[:, :, :, i],
                                                                                      kernel[:, :, :,
                                                                                      i + 1], stride)
        write_experimental_data_to_excel(worksheet, layer_num, (i, i + 1), number_of_memory_access_weights,
                                         number_of_memory_access_input,
                                         number_of_access_to_multiplier, number_of_access_to_accumulator2,
                                         number_of_access_to_accumulator3,
                                         wiT1_table_size, wiT2_table_size, input_table_size, access_input_buffer,
                                         access_weight_buffer, access_partial_sum_buffer)
    
    workbook.close()
    return result + bias

def write_experimental_data_to_excel(worksheet, layer_num, filter_num, n_memory_access_weights, n_memory_access_inputs,
                                     n_access_to_multiplier,
                                     n_access_to_accumulator2, n_access_to_accumulator3, size_of_wiT1, size_of_wiT2,
                                     size_of_input_table, access_input_buffer, access_weight_buffer,
                                     access_partial_sum_buffer):
    if filter_num == (0, 1):
        worksheet.write('A1', 'Filter')
        worksheet.write('B1', '#memory_access_weights')
        worksheet.write('C1', '#memory_access_inputs')
        worksheet.write('D1', '#access_multiplier')
        worksheet.write('E1', '#access_accumulator2')
        worksheet.write('F1', '#access_accumulator3')
        worksheet.write('G1', 'size_of_wiT1')
        worksheet.write('H1', 'size_of_wiT2')
        worksheet.write('I1', 'size_of_input_table')
        worksheet.write('J1', '#access_input_buffer')
        worksheet.write('K1', '#access_weight_buffer')
        worksheet.write('L1', '#access_partial_sum_buffer')

    worksheet.write('A' + str(filter_num[0] / 2 + 2), f'Filter{filter_num}')
    worksheet.write('B' + str(filter_num[0] / 2 + 2), n_memory_access_weights)
    worksheet.write('C' + str(filter_num[0] / 2 + 2), n_memory_access_inputs)
    worksheet.write('D' + str(filter_num[0] / 2 + 2), n_access_to_multiplier)
    worksheet.write('E' + str(filter_num[0] / 2 + 2), n_access_to_accumulator2)
    worksheet.write('F' + str(filter_num[0] / 2 + 2), n_access_to_accumulator3)
    worksheet.write('G' + str(filter_num[0] / 2 + 2), size_of_wiT1)
    worksheet.write('H' + str(filter_num[0] / 2 + 2), size_of_wiT2)
    worksheet.write('I' + str(filter_num[0] / 2 + 2), size_of_input_table)
    worksheet.write('J' + str(filter_num[0] / 2 + 2), access_input_buffer)
    worksheet.write('K' + str(filter_num[0] / 2 + 2), access_weight_buffer)
    worksheet.write('L' + str(filter_num[0] / 2 + 2), access_partial_sum_buffer)
