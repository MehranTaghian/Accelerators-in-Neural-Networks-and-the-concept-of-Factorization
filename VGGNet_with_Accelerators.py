from Accelerator_cython import factorization
import numpy as np
import tensorflow as tf

sess = tf.Session()

input_image = np.random.random([224, 224, 3])

# net_data = np.load(open(r"C:\Users\Mehran\Desktop\Lotfi-Kamran\Weights\bvlc_alexnet.npy", "rb"), encoding="latin1",
#                    allow_pickle=True).item()


def conv_layer(input_data, key, stride=1, padding="VALID"):
    convW = np.load(open("..\\Weights\\VGG\\" + key + "_W.npy", "rb"), encoding="latin1")
    convb = np.load(open("..\\Weights\\VGG\\" + key + "_b.npy", "rb"), encoding="latin1")

    # test

    # convW = np.random.randint(20, size=convW.shape)
    # convb = np.random.randint(20, size=convb.shape)
    # convW = np.ones(convW.shape)
    # convb = np.ones(convb.shape)

    #

    conv_temp = factorization.convolve(input_data, convW, convb, layer_num=key, stride=stride, padding=padding)
    conv_relu = tf.nn.relu(conv_temp)
    conv = sess.run(conv_relu)
    return conv


print('layer1')

# layer 1 convolution
conv1_1 = conv_layer(input_image, "conv1_1", padding="SAME")
print(conv1_1.shape)
conv1_2 = conv_layer(conv1_1, "conv1_2", padding="SAME")
print(conv1_2.shape)
# max-pooling layer1
maxpool1_temp = tf.nn.max_pool(tf.reshape(conv1_2, [1, conv1_2.shape[0], conv1_2.shape[1], conv1_2.shape[2]]),
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
maxpool1 = sess.run(maxpool1_temp)
maxpool1 = maxpool1.reshape([maxpool1.shape[1], maxpool1.shape[2], maxpool1.shape[3]])
print(maxpool1.shape)

print('layer2')
# layer 2 convolution
conv2_1 = conv_layer(maxpool1, "conv2_1", padding="SAME")
print(conv2_1.shape)
conv2_2 = conv_layer(conv2_1, "conv2_2", padding="SAME")
print(conv2_2.shape)

# max-pooling layer2
maxpool2_temp = tf.nn.max_pool(tf.reshape(conv2_2, [1, conv2_2.shape[0], conv2_2.shape[1], conv2_2.shape[2]]),
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
maxpool2 = sess.run(maxpool2_temp)
maxpool2 = maxpool2.reshape([maxpool2.shape[1], maxpool2.shape[2], maxpool2.shape[3]])


print('layer3')
# layer 3 convolution
conv3_1 = conv_layer(maxpool2, "conv3_1", padding="SAME")
print(conv3_1.shape)
conv3_2 = conv_layer(conv3_1, "conv3_2", padding="SAME")
print(conv3_2.shape)
conv3_3 = conv_layer(conv3_2, "conv3_3", padding="SAME")
print(conv3_3.shape)

# max-pooling layer3
maxpool3_temp = tf.nn.max_pool(tf.reshape(conv3_3, [1, conv3_3.shape[0], conv3_3.shape[1], conv3_3.shape[2]]),
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
maxpool3 = sess.run(maxpool3_temp)
maxpool3 = maxpool3.reshape([maxpool3.shape[1], maxpool3.shape[2], maxpool3.shape[3]])

print('layer4')
# layer 4 convolution
conv4_1 = conv_layer(maxpool3, "conv4_1", padding="SAME")
print(conv4_1.shape)
conv4_2 = conv_layer(conv4_1, "conv4_2", padding="SAME")
print(conv4_2.shape)
conv4_3 = conv_layer(conv4_2, "conv4_3", padding="SAME")
print(conv4_3.shape)

# max-pooling layer4
maxpool4_temp = tf.nn.max_pool(tf.reshape(conv4_3, [1, conv4_3.shape[0], conv4_3.shape[1], conv4_3.shape[2]]),
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
maxpool4 = sess.run(maxpool4_temp)
maxpool4 = maxpool4.reshape([maxpool4.shape[1], maxpool4.shape[2], maxpool4.shape[3]])

print('layer5')
# layer 5 convolution
conv5_1 = conv_layer(maxpool4, "conv5_1", padding="SAME")
print(conv5_1.shape)
conv5_2 = conv_layer(conv5_1, "conv5_2", padding="SAME")
print(conv5_2.shape)
conv5_3 = conv_layer(conv5_2, "conv5_3", padding="SAME")
print(conv5_3.shape)

# max-pooling layer5
maxpool5_temp = tf.nn.max_pool(tf.reshape(conv5_3, [1, conv5_3.shape[0], conv5_3.shape[1], conv5_3.shape[2]]),
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
maxpool5 = sess.run(maxpool5_temp)
maxpool5 = maxpool5.reshape([maxpool5.shape[1], maxpool5.shape[2], maxpool5.shape[3]])




# print('layer2')
#
# print('layer3')
#
# print('layer4')
# print('layer5')
# # layer 5 convolution
# conv5 = conv_layer(conv4, "conv5", padding="SAME")
#
# # max-pooling layer5
# maxpool5_temp = tf.nn.max_pool(tf.reshape(conv5, [1, conv5.shape[0], conv5.shape[1], conv5.shape[2]]),
#                                ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
# maxpool5 = sess.run(maxpool5_temp)
# maxpool5 = maxpool5.reshape([maxpool5.shape[1], maxpool5.shape[2], maxpool5.shape[3]])
#
# # layer 6 fc
# print('layer6')
# fc6 = maxpool5.reshape([9216])
#
# # layer 7 fc
# print('layer7')
# fc6W = np.array(net_data["fc6"][0])
# fc6b = np.array(net_data["fc6"][1])
# fc7 = fc6.dot(fc6W) + fc6b
#
# # layer 8 fc
# print('layer8')
# fc7W = np.array(net_data["fc7"][0])
# fc7b = np.array(net_data["fc7"][1])
# fc8 = fc7.dot(fc7W) + fc7b
#
# # output
# print('output')
# fc8W = np.array(net_data["fc8"][0])
# fc8b = np.array(net_data["fc8"][1])
# output = fc8.dot(fc8W) + fc8b
