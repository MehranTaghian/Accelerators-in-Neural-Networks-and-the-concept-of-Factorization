import cython
import numpy as np
cimport numpy as np
from UCNN_lane import conv_single_stride

@cython.cdivision(True)
cpdef conv2d(np.ndarray data, np.ndarray filter1, np.ndarray filter2, int stride=1):
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
        int result_size = int((data.shape[0] - filter1.shape[0]) / stride + 1)
        np.ndarray result = np.zeros([result_size, result_size, 2])
        int width = ((data.shape[1] - filter1.shape[1]) / stride) + 1
        int height = ((data.shape[0] - filter1.shape[0]) / stride) + 1
        int i = 0, j = 0, key = 0
    #This loop is for result
    # while i < height:
    #     j = 0
    #     while j < width:
    #         temp_result = 0
    #         ind = 0
    #         while ind < weights.shape[0]:

    while i < (result.shape[0]):
        j = 0
        while j < (result.shape[1]):
            # print(data[i: i + filter1.shape[0], j: j + filter1.shape[1], :])
            result[i, j, :] = conv_single_stride(2, filter1, filter2, data[i: i + filter1.shape[0], j: j + filter1.shape[1], :])

            j += stride
        i += stride

        #this part for i and j are for result
        #     j += 1
        # i += 1
    return result

