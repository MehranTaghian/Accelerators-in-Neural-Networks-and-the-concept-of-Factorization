import numpy as np
from PE import PE


class ExperimentalParameters:
    def __init__(self):
        self.dict_params = {'size_of_factorized_storage': 0,
                            # 'size_of_global_buffer': 0,
                            'utilized_size_of_factorized_storage': 0,
                            # 'utilized_size_of_global_buffer': 0,
                            'number_of_access_to_pu': 0,
                            'number_of_addition': 0,
                            'number_of_multiply': 0,
                            'number_of_access_to_cu': 0,
                            'number_of_access_to_global_buffer': 0,
                            'number_of_issuing_inputs_to_pe': 0,
                            'number_of_issuing_weights_to_pe': 0,
                            'number_of_write_data_to_global_buffer': 0,
                            'number_of_read_non_zero_data_from_global_buffer': 0}


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
    def __init__(self, inputs, kernel, exp_params: ExperimentalParameters, buss_size=8):
        """

        :param inputs: input data to CU
        :param kernel: input weights to CU
        :param exp_params: Experimental Parameters
        """

        unique_weights, repeated = conv_factorization(kernel)
        self.unique_weights = unique_weights.tolist()
        self.inputs_for_unique_weights = {}
        factorized_storage_size_temp = 0
        utilized_factorized_storage_size_temp = 0
        self.number_of_access_to_global_buffer = 0
        for i in range(len(self.unique_weights)):
            if self.unique_weights[i] != 0:
                factorized_storage_size_temp += repeated[i].shape[0]
                utilized_factorized_storage_size_temp += repeated[i].shape[0] - np.where(
                    inputs[repeated[i][:, 0], repeated[i][:, 1], repeated[i][:, 2]] == 0)[0].shape[0]

                exp_params.dict_params['number_of_access_to_global_buffer'] += repeated[i].shape[0]

                exp_params.dict_params[
                    'number_of_read_non_zero_data_from_global_buffer'] += repeated[i].shape[0] - np.where(
                    inputs[repeated[i][:, 0], repeated[i][:, 1], repeated[i][:, 2]] == 0)[0].shape[0]

                self.inputs_for_unique_weights[self.unique_weights[i]] = inputs[
                    repeated[i][:, 0], repeated[i][:, 1], repeated[i][:, 2]]

        self.unique_weights.remove(0)

        exp_params.dict_params['size_of_factorized_storage'] = factorized_storage_size_temp \
            if factorized_storage_size_temp > exp_params.dict_params['size_of_factorized_storage'] \
            else exp_params.dict_params['size_of_factorized_storage']

        exp_params.dict_params['utilized_size_of_factorized_storage'] = utilized_factorized_storage_size_temp \
            if utilized_factorized_storage_size_temp > exp_params.dict_params['utilized_size_of_factorized_storage'] \
            else exp_params.dict_params['utilized_size_of_factorized_storage']


def conv_single_stride(inputs, filter, exp_params: ExperimentalParameters):
    """

    :param inputs: inputs for one stride
    :param filter: specific channel
    :param exp_params: Experimental Parameters
    :return:
    """
    cu = CU(inputs, filter, exp_params)
    global_8bit_adder = 0
    exp_params.dict_params['number_of_access_to_cu'] += 1

    for weight in cu.unique_weights:
        pe = PE(weight)
        exp_params.dict_params['number_of_access_to_pu'] += 1
        i = 0
        while i < len(cu.inputs_for_unique_weights[weight]):
            number_of_inputs_to_pe = 8 if i + 8 <= len(cu.inputs_for_unique_weights[weight]) else len(
                cu.inputs_for_unique_weights[weight]) - i
            pe.parallel_8bit_adder(cu.inputs_for_unique_weights[weight][i:i + number_of_inputs_to_pe])
            i += 8
            exp_params.dict_params['number_of_access_to_pu'] += number_of_inputs_to_pe
            exp_params.dict_params['number_of_access_to_cu'] += number_of_inputs_to_pe
            exp_params.dict_params['number_of_issuing_inputs_to_pe'] += number_of_inputs_to_pe

        exp_params.dict_params['number_of_addition'] += pe.num_addition
        exp_params.dict_params['number_of_multiply'] += 1
        exp_params.dict_params['number_of_access_to_pu'] += 1
        exp_params.dict_params['number_of_access_to_cu'] += 1
        global_8bit_adder += pe.psum * weight
        exp_params.dict_params['number_of_issuing_weights_to_pe'] += 1

    return global_8bit_adder


def conv2d(data, filter, exp_params, stride=1):
    """
        The kernel is 3d like in image with 3 channels RGB
    :param exp_params: Experimental Parameters
    :param filter: input filter of a channel
    :param data: is the actual input to convolution
    :param stride: steps of moving kernel
    :return: the result of convolution
    """
    result_size = int((data.shape[0] - filter.shape[0]) / stride + 1)
    result = np.zeros([result_size, result_size])
    i = 0
    j = 0
    while i < (result.shape[0]):
        j = 0
        while j < (result.shape[1]):
            row_index, col_index = i * stride, j * stride
            result[i, j] = conv_single_stride(
                data[row_index: row_index + filter.shape[0], col_index: col_index + filter.shape[1], :], filter,
                exp_params)

            if result[i, j] != 0:
                exp_params.dict_params['number_of_write_data_to_global_buffer'] += 1

            j += 1
        i += 1
    return result


def convolution(worksheet, data, kernel, bias, layer_num, stride=1, padding="VALID"):
    """
    :param worksheet: For saving the result data into an excel file
    :param data: input data for convolution
    :param kernel:
    :param bias: bias of each neuron
    :param layer_num: which layer of the Deep Neural Network(say alexnet)
    :param stride: number of stride
    :param padding:number of padding
    :return: result of the convolution
    """
    exp = ExperimentalParameters()
    exp.size_of_global_buffer = data.size
    exp.utilized_size_of_global_buffer = data.size - np.where(data == 0)[0].shape[0]

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
    for i in range(0, filter_num):
        result[:, :, i] = conv2d(input_data_padded,
                                 kernel[:, :, :, i],
                                 exp, stride)

    write_experimental_data_to_excel(worksheet, layer_num, exp)
    return result + bias


def write_experimental_data_to_excel(worksheet, layer_num, exp_params):
    if layer_num == 1:
        i = 65
        for k in exp_params.dict_params.keys():
            worksheet[f'{chr(i)}1'] = k
            i += 1

    i = 65
    for k, v in exp_params.dict_params.items():
        worksheet[f'{chr(i)}{layer_num + 1}'] = v
        i += 1

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
