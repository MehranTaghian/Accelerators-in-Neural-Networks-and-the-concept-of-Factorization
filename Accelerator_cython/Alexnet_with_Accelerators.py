import numpy as np
import tensorflow as tf
import factorization
from Get_alexnet_weights import get_weights

input_image = np.load('test_image.npy')
input_image = tf.constant(input_image, dtype=tf.float32)
input_image = tf.reshape(input_image, [1, input_image.shape[0], input_image.shape[1], input_image.shape[2]])
sess = tf.Session()
input_image = np.array(sess.run(input_image))
# net_data = np.load(open(r"C:\Users\Mehran\Desktop\Lotfi-Kamran\Weights\bvlc_alexnet.npy", "rb"), encoding="latin1",
#                    allow_pickle=True).item()

# net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1",
#                    allow_pickle=True).item()
# weight, bias = get_weights('../alex_net')

net_data = np.load(open(r"C:\Users\Mehran\Desktop\Lotfi-Kamran\Weights\AlexNet_quantized_weights_4.pickle", "rb"),
                   encoding="latin1",
                   allow_pickle=True)


def conv_layer(input_data, key, group=1, stride=1, padding="VALID"):
    # convW = np.array(net_data[key][0])
    # convb = np.array(net_data[key][1])
    # convW = np.array(weight[f'conv{key}'], dtype=np.float32)
    # convb = np.array(bias[f'b{key}'], dtype=np.float32)
    # convW = convW.reshape([convW.shape[2], convW.shape[3], convW.shape[1], convW.shape[0]])
    # print(convW.shape)

    convW = np.array(net_data[f'convolution_{key}'], dtype=np.float32)
    convb = np.array(net_data[f'bias_{key}'], dtype=np.float32)

    # input_data_numpy = np.array(input_data.eval(session=sess))
    input_data_factorization = input_data.copy()
    input_data_factorization = input_data_factorization.reshape(
        [input_data.shape[1], input_data.shape[2], input_data.shape[3]])
    factorization.convolve(input_data_factorization, convW, convb, layer_num=f'convolution_{key}', stride=stride,
                           padding=padding)

    kernel = tf.constant(convW, tf.float32)
    biases = tf.constant(convb, tf.float32)
    temp1 = tf.nn.convolution(input_data, kernel, padding, [1, stride, stride, 1])
    temp2 = tf.nn.relu(temp1)
    result = tf.nn.bias_add(temp2, biases)
    return sess.run(result)
    # c_i = kernel.get_shape()[-1]
    # c_o = convW.shape[0]
    # assert c_i % group == 0
    # assert c_o % group == 0
    # convolve = lambda i, k: tf.nn.conv2d(i, k, [1, stride, stride, 1], padding=padding)
    #
    # if group == 1:
    #     conv = convolve(input_data, kernel)
    # else:
    #     input_groups = tf.split(input_data, group, 3)  # tf.split(3, group, input)
    #     kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
    #     output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
    #     conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
    # return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


print('layer1')
# layer 1 convolution
# conv1 = conv_layer(input_image, "conv1", stride=4)
conv1 = conv_layer(input_image, 1, group=1, stride=4, padding='SAME')
# max-pooling layer1
print('conv1.shape', conv1.shape)
maxpool1 = tf.nn.max_pool(conv1,
                          ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
maxpool1 = sess.run(maxpool1)
# maxpool1 = maxpool1.reshape([maxpool1.shape[1], maxpool1.shape[2], maxpool1.shape[3]])
#

print('pool1.shape', maxpool1.shape)
print('layer2')

# layer 2 convolution
# conv2 = conv_layer(maxpool1, "conv2", padding="SAME")
conv2 = conv_layer(maxpool1, 2, group=2, padding="SAME")  # max-pooling layer2
maxpool2 = tf.nn.max_pool(conv2,
                          ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
maxpool2 = sess.run(maxpool2)

print('conv2.shape', conv2.shape)
print('maxpool2.shape', maxpool2.shape)
# maxpool2 = maxpool2.reshape([maxpool2.shape[1], maxpool2.shape[2], maxpool2.shape[3]])

print('pool2.shape', maxpool2.shape)
print('layer3')

# layer 3 convolution
# conv3 = conv_layer(maxpool2, "conv3", padding="SAME")
conv3 = conv_layer(maxpool2, 3, padding="SAME")
print('layer4')
# layer 4 convolution
# conv4 = conv_layer(conv3, "conv4", padding="SAME")

print('conv3.shape', conv3.shape)
conv4 = conv_layer(conv3, 4, group=2, padding="SAME")
print('layer5')
# layer 5 convolution
# conv5 = conv_layer(conv4, "conv5", padding="SAME")

print('conv4.shape', conv4.shape)
conv5 = conv_layer(conv4, 5, group=2, padding="SAME")
# max-pooling layer5
maxpool5_temp = tf.nn.max_pool(conv5,
                               ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
maxpool5 = sess.run(maxpool5_temp)
maxpool5 = maxpool5.reshape([maxpool5.shape[1], maxpool5.shape[2], maxpool5.shape[3]])

print('conv5.shape', conv5.shape)
print('pool5.shape', maxpool5.shape)
# layer 6 fc
print('layer6')
fc6 = maxpool5.reshape(-1)

# layer 7 fc
print('layer7')
# fc6W = np.array(net_data["fc6"][0])
# fc6b = np.array(net_data["fc6"][1])
fc6W = np.array(net_data["fc_1"].T)
fc6b = np.array(net_data["fc_1_bias"])
fc7 = fc6.dot(fc6W) + fc6b

# layer 8 fc
print('layer8')
# fc7W = np.array(net_data["fc7"][0])
# fc7b = np.array(net_data["fc7"][1])
fc7W = np.array(net_data["fc_2"].T)
fc7b = np.array(net_data["fc_2_bias"])
fc8 = fc7.dot(fc7W) + fc7b

# output
print('output')
fc8W = np.array(net_data["fc_3"].T)
fc8b = np.array(net_data["fc_3_bias"])
output = fc8.dot(fc8W) + fc8b

# # layer 7 fc
# print('layer7')
# # fc6W = np.array(net_data["fc6"][0])
# # fc6b = np.array(net_data["fc6"][1])
# fc6W = np.array(weight["fc6"].T)
# fc6b = np.array(bias["b6"])
# fc7 = fc6.dot(fc6W) + fc6b
#
# # layer 8 fc
# print('layer8')
# # fc7W = np.array(net_data["fc7"][0])
# # fc7b = np.array(net_data["fc7"][1])
# fc7W = np.array(weight["fc7"].T)
# fc7b = np.array(bias["b7"])
# fc8 = fc7.dot(fc7W) + fc7b
#
# # output
# print('output')
# fc8W = np.array(weight["fc8"].T)
# fc8b = np.array(bias["b8"])
# output = fc8.dot(fc8W) + fc8b
