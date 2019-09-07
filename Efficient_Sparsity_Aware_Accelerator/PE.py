import numpy as np


class PE:

    def __init__(self, weight):
        self.weight = weight
        self.psum = 0
        self.num_addition = 0

    def parallel_8bit_adder(self, inputs, num_inputs_elements=8):
        """

        :param inputs: A numpy.ndarray with 8 elements of qint-8bit
        """
        self.psum += np.sum(inputs)
        self.num_addition += num_inputs_elements
