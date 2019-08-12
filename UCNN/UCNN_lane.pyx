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

    def partial_sum2(self, a):
        self.accumulator2 += a

    def partial_sum3(self, a):
        self.accumulator3 += a

    def send_to_accumulator3(self):
        self.accumulator3 += self.accumulator2

    def multiplier(self, weight, which_kernel: int):
        """

        :param weight:
        :param which_kernel: 1 for kernel wiT1 and 2 for kernel wiT2
        :return:
        """
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
    # print('indices1:', indices[0])
    # print('inputs before', inputs)
    # print(weights)
    # print(indices)
    weights, inputs, indices, len_non_zeros = remove_zero_weights(weights, inputs, indices)
    # print('inputs after', inputs)
    # print(weights)
    # print(indices)

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
        # if temp.shape[0] != 0:
        #     index = temp[0]
        # else:
        #     i += 1
        #     continue
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
    # remaining_weights = []
    # while i < inputs2.shape[0]:
    #     if flag[i] == 0:  # meaning that it wasn't added in the previous loop
    #         new_inputs2[counter] = inputs2[i]
    #         counter += 1
    #         j = 0
    #         while i >= indices[j]:
    #             j += 1
    #         # wiT2[counter] = weights[j]
    #         remaining_weights.append(weights[j])
    #     i += 1
    return np.array(wiT2), np.array(sorted_weights)  #, np.array(remaining_weights), new_inputs2

cpdef conv_single_stride(int g, np.ndarray filter1, np.ndarray filter2, np.ndarray input):
    wiT1, positions1 = get_wiT1(filter1)
    wiT2, sorted_weights2 = get_wiT2(filter2, positions1)
    pe = PE()
    cdef float result1 = 0, result2 = 0

    #-----------------------------alaki
    # cdef int j = 0
    # print(wiT1)
    # while j < wiT1.shape[0]:
    #     temp = 0
    #     while wiT1[j] == 0:
    #         # print(input[positions1[j][0], positions1[j][1], positions1[j][2]])
    #         temp += input[positions1[j][0], positions1[j][1], positions1[j][2]]
    #         j += 1
    #     temp += input[positions1[j][0], positions1[j][1], positions1[j][2]]
    #     result1 += temp * wiT1[j]
    #     j += 1
    # print(result1)
    # result1 = 0
    # j = 0
    # print(wiT2)
    # while j < wiT2.shape[0]:
    #     temp = 0
    #     while wiT2[j] == 0:
    #         # print([positions2[j][0], positions2[j][1], positions2[j][2]])
    #         temp += input[positions2[j][0], positions2[j][1], positions2[j][2]]
    #         j += 1
    #     # print([positions2[j][0], positions2[j][1], positions2[j][2]])
    #     temp += input[positions2[j][0], positions2[j][1], positions2[j][2]]
    #     result2 += temp * wiT2[j]
    #     j += 1
    # print(result2)
    # result1 = 0
    # result2 = 0
    # ---------------------------------------
    # print('wiT1', wiT1)
    cdef int i = 0
    # size = wiT1.shape[0] if wiT1.shape[0] < wiT2.shape[0] else wiT2.shape[0]
    weight2_index = 0
    while i < wiT1.shape[0]:

        if wiT1[i] == 0 and wiT2[i] == 0:
            pe.partial_sum2(input[positions1[i][0], positions1[i][1], positions1[i][2]])

        elif wiT1[i] != 0 and wiT2[i] == 0:
            pe.partial_sum2(input[positions1[i][0], positions1[i][1], positions1[i][2]])
            result1 += pe.multiplier(wiT1[i], 1)
            pe.use_accumulator3 = 2
            pe.send_to_accumulator3()
            pe.accumulator2 = 0

        elif wiT2[i] != 0 and wiT1[i] == 0:
            pe.partial_sum2(input[positions1[i][0], positions1[i][1], positions1[i][2]])
            if sorted_weights2[weight2_index] != -1:
                result2 += pe.multiplier(sorted_weights2[weight2_index], 2)
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
            result1 += pe.multiplier(wiT1[i], 1)
            if sorted_weights2[weight2_index] != -1:
                result2 += pe.multiplier(sorted_weights2[weight2_index], 2)
            else:
                if pe.use_accumulator3 == 2:
                    pe.accumulator3 = 0
                    pe.use_accumulator3 = 0
            weight2_index += 1
            pe.accumulator2 = 0

        i += 1
    while i < wiT2.shape[0]:
        if wiT2[i] == 0:
            if sorted_weights2[weight2_index] != -1:  #weight is not zero
                pe.partial_sum2(input[positions1[i][0], positions1[i][1], positions1[i][2]])
        else:
            if sorted_weights2[weight2_index] != -1:  #weight is not zero
                pe.partial_sum2(input[positions1[i][0], positions1[i][1], positions1[i][2]])
                result2 += pe.multiplier(sorted_weights2[weight2_index], 2)
            weight2_index += 1
            pe.use_accumulator3 = 0
            pe.accumulator2 = 0
        i += 1

    # if size == wiT1.shape[0]:
    #     print('yes')
    #     while i < wiT2.shape[0]:s
    #         temp = input[positions2[i][0], positions2[i][1], positions2[i][2]]
    #         if pe.use_accumulator3 == 2:
    #             temp += pe.accumulator3
    #             pe.accumulator3 = 0
    #             pe.use_accumulator3 = 0
    #         result2 += wiT2[i] * temp
    #         i += 1
    # else:
    #     while i < positions1.shape[0]:
    #         pe.partial_sum2(input[positions1[i][0], positions1[i][1], positions1[i][2]])
    #         while wiT1[i] == 0:
    #             i += 1
    #             pe.partial_sum2(input[positions1[i][0], positions1[i][1], positions1[i][2]])
    #         result1 += pe.multiplier(wiT1[i], 1)
    #         pe.accumulator2 = 0
    #         i += 1
    # print(positions1)
    # print(remaining_wiT2)
    # print(positions2)
    # if remaining_wiT2.shape[0] > 0:
    #     counter = wiT2.shape[0]
    #     i = 0
    #     while i < remaining_wiT2.shape[0]:
    #         result2 += remaining_wiT2[i] * input[positions2[counter][0], positions2[counter][1], positions2[counter][2]]
    #         counter += 1
    #         print(counter < positions2.shape[0])
    #         i += 1
    result = np.zeros(2)
    result[0] = result1
    result[1] = result2
    return result
