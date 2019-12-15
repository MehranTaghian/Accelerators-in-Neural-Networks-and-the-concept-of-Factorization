from Factorization import *
import numpy as np
cimport numpy as np


class PE:
    """
    use_accumulator3: set to 1 if wiT1's sum value stored in it else if 2
                        is using it, set to 2, else 0
    """

    def __init__(self):
        self.accumulator2 = 0
        self.accumulator3 = 0
        self.use_accumulator3 = 0
        self.number_of_access_to_multiplier = 0
        self.number_of_access_to_accumulator2 = 0
        self.number_of_access_to_accumulator3 = 0

    def partial_sum2(self, a):
        self.number_of_access_to_accumulator2 += 1
        self.accumulator2 += a

    def partial_sum3(self, a):
        self.number_of_access_to_accumulator3 += 1
        self.accumulator3 += a

    def send_to_accumulator3(self):
        self.number_of_access_to_accumulator3 += 1
        self.accumulator3 += self.accumulator2

    def multiplier(self, weight, which_kernel: int):
        """

        :param weight:
        :param which_kernel: 1 for kernel wiT1 and 2 for kernel wiT2
        :return:
        """
        self.number_of_access_to_multiplier += 1
        if which_kernel == 1:
            if self.use_accumulator3 == 1:
                result = weight * (self.accumulator2 + self.accumulator3)
                self.accumulator3 = 0
                self.use_accumulator3 = 0
            else:
                result = weight * self.accumulator2
        elif which_kernel == 2:
            if self.use_accumulator3 == 2:
                result = weight * (self.accumulator2 + self.accumulator3)
                self.accumulator3 = 0
                self.use_accumulator3 = 0
            else:
                result = weight * self.accumulator2
        return result


def remove_zero_weights(weights, inputs, indices):
    """
    :param weights: weights of the kernel
    :param inputs: numbers in the input data
    :param indices: indices for each weight that would be multiplied to the input data
    :return: weights and inputs and indices without zero weights
    """
    if np.where(weights == 0)[0].shape[0] > 0:
        index = int(np.where(weights == 0)[0])
        new_indices = np.zeros(indices.shape[0] - 1, dtype=int)
        new_inputs = np.zeros(inputs.shape, dtype=int)

        len_removed = indices[index] - (indices[index - 1] if index > 0 else 0)
        new_inputs[0:inputs.shape[0] - len_removed] = np.delete(inputs, range(indices[index - 1] if index > 0 else 0,
                                                                              indices[index]), axis=0)
        new_inputs[inputs.shape[0] - len_removed: inputs.shape[0]] = inputs[
                                                                     indices[index - 1] if index > 0 else 0: indices[
                                                                         index]]
        i = 0
        while i < index:
            new_indices[i] = indices[i]
            i += 1
        i += 1
        for j in range(i - 1, new_indices.shape[0]):
            new_indices[j] = indices[i] - len_removed
            i += 1
        weights = np.delete(weights, np.where(weights == 0))
        return weights, new_inputs, new_indices, (inputs.shape[0] - len_removed)
    return weights, inputs, indices, 0

def get_wiT1(kernel):
    # print(kernel)
    weights, inputs, indices = conv_factorization(kernel)
    weights, inputs, indices, len_non_zeros = remove_zero_weights(weights, inputs, indices)

    wiT1 = np.zeros(len_non_zeros, dtype=int)
    wiT1[indices - 1] = weights
    return wiT1, inputs

def get_wiT2(kernel, inputs):
    weights, inputs2, indices = conv_factorization(kernel)
    # weights, inputs2, indices = remove_zero_weights(weights, inputs2, indices)
    wiT2 = np.zeros(inputs.shape[0], dtype=int)
    sorted_weights = []
    i = 0
    w_prev = -1000
    new_inputs2 = np.zeros(inputs2.shape, dtype=int)
    flag = np.zeros(inputs2.shape[0])
    counter = 0
    # size = inputs.shape[0] if inputs.shape[0] < inputs2.shape[0] else inputs2.shape[0]
    while i < inputs.shape[0]:
        # flag += np.all(inputs2 == inputs[i], axis=1)
        temp = np.where(np.all(inputs2 == inputs[i], axis=1))[0]
        index = temp[0]
        j = 0
        while index >= indices[j]:
            j += 1
        w_new = weights[j]
        # wiT2[counter] = w_new
        wiT2[i] = 1
        # counter += 1
        # new_inputs2[counter] = inputs[i]
        if w_prev == w_new:
            wiT2[i - 1] = 0
        else:
            if w_new != 0:
                sorted_weights.append(w_new)
            else:
                sorted_weights.append(-1)

        w_prev = w_new
        i += 1
    i = 0
    j = 0
    return np.array(wiT2), np.array(sorted_weights)  #, np.array(remaining_weights), new_inputs2

cpdef conv_single_stride(np.ndarray filter1, np.ndarray filter2, np.ndarray input):
    """
    This function takes 2 filters and calculates the result of a single stride in convolving
    :param filter1: First kernel
    :param filter2: Second Kernel
    :param input: input data
    :return: output of the convolution
    """
    wiT1, positions1 = get_wiT1(filter1)
    wiT2, sorted_weights2 = get_wiT2(filter2, positions1)
    pe = PE()
    cdef float result1 = 0, result2 = 0
    cdef int number_of_memory_access_input = 0, number_of_memory_access_weights = 0
    cdef int wiT1_table_size = wiT1.shape[0], wiT2_table_size = wiT2.shape[0], input_table_size = positions1.shape[0]
    cdef int access_input_buffer = 0, access_weight_buffer = 0, access_partial_sum_buffer = 0

    cdef int i = 0
    # size = wiT1.shape[0] if wiT1.shape[0] < wiT2.shape[0] else wiT2.shape[0]
    weight2_index = 0
    while i < wiT1.shape[0]:

        access_weight_buffer += 2
        access_input_buffer += 1

        if wiT1[i] == 0 and wiT2[i] == 0:
            pe.partial_sum2(input[positions1[i][0], positions1[i][1], positions1[i][2]])
            number_of_memory_access_input += 1

        elif wiT1[i] != 0 and wiT2[i] == 0:
            pe.partial_sum2(input[positions1[i][0], positions1[i][1], positions1[i][2]])
            number_of_memory_access_input += 1

            result1 += pe.multiplier(wiT1[i], 1)
            access_partial_sum_buffer += 1

            number_of_memory_access_weights += 1
            pe.use_accumulator3 = 2
            pe.send_to_accumulator3()
            pe.accumulator2 = 0

        elif wiT2[i] != 0 and wiT1[i] == 0:
            pe.partial_sum2(input[positions1[i][0], positions1[i][1], positions1[i][2]])
            if sorted_weights2[weight2_index] != -1:
                result2 += pe.multiplier(sorted_weights2[weight2_index], 2)
                access_partial_sum_buffer += 1

                number_of_memory_access_weights += 1
            else:
                if pe.use_accumulator3 == 2:
                    pe.accumulator3 = 0
                    pe.use_accumulator3 = 0

            weight2_index += 1
            pe.use_accumulator3 = 1
            pe.send_to_accumulator3()
            pe.accumulator2 = 0

        elif wiT1[i] != 0 and wiT2[i] != 0:
            pe.partial_sum2(input[positions1[i][0], positions1[i][1], positions1[i][2]])
            number_of_memory_access_input += 1

            result1 += pe.multiplier(wiT1[i], 1)
            access_partial_sum_buffer += 1

            number_of_memory_access_weights += 1
            if sorted_weights2[weight2_index] != -1:
                result2 += pe.multiplier(sorted_weights2[weight2_index], 2)
                access_partial_sum_buffer += 1

                number_of_memory_access_weights += 1
            else:
                if pe.use_accumulator3 == 2:
                    pe.accumulator3 = 0
                    pe.use_accumulator3 = 0
            weight2_index += 1
            pe.accumulator2 = 0

        i += 1
    while i < wiT2.shape[0]:
        access_weight_buffer += 1
        access_input_buffer += 1

        if wiT2[i] == 0:
            if sorted_weights2[weight2_index] != -1:  #weight is not zero
                pe.partial_sum2(input[positions1[i][0], positions1[i][1], positions1[i][2]])
                number_of_memory_access_input += 1
        else:
            if sorted_weights2[weight2_index] != -1:  #weight is not zero
                pe.partial_sum2(input[positions1[i][0], positions1[i][1], positions1[i][2]])

                result2 += pe.multiplier(sorted_weights2[weight2_index], 2)
                access_partial_sum_buffer += 1

                number_of_memory_access_input += 1
                number_of_memory_access_weights += 1
            weight2_index += 1
            pe.use_accumulator3 = 0
            pe.accumulator2 = 0
        i += 1

    result = np.zeros(2)
    result[0] = result1
    result[1] = result2
    return result, number_of_memory_access_weights, number_of_memory_access_input, \
           pe.number_of_access_to_accumulator2, pe.number_of_access_to_accumulator3, \
           pe.number_of_access_to_multiplier, wiT1_table_size, wiT2_table_size, input_table_size, \
           access_input_buffer, access_weight_buffer, access_partial_sum_buffer
