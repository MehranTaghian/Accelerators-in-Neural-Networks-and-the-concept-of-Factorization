import numpy as np
import tensorflow as tf
import factorization
from Get_alexnet_weights import get_weights
import xlsxwriter as excel

input_image = np.load('test_image.npy')
input_image = tf.constant(input_image, dtype=tf.float32)
input_image = tf.reshape(input_image, [1, input_image.shape[0], input_image.shape[1], input_image.shape[2]])
sess = tf.Session()
input_image = np.array(sess.run(input_image))
# net_data = np.load(open(r"C:\Users\Mehran\Desktop\Lotfi-Kamran\Weights\bvlc_alexnet.npy", "rb"), encoding="latin1",
#                    allow_pickle=True).item()

# net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1",
#                    allow_pickle=True).item()
weight, bias = get_weights('../alex_net')

net_data = np.load(open(r"C:\Users\Mehran\Desktop\Lotfi-Kamran\Weights\AlexNet_quantized_weights_4.pickle", "rb"),
                   encoding="latin1",
                   allow_pickle=True)


def conv_layer(input_data_non_quantized, input_data_quantized, key, stride=1, padding="VALID"):
    if input_data_quantized is None:
        input_data_quantized = input_data_non_quantized

    convW_non_quantized = np.array(weight[f'conv{key}'], dtype=np.float32)
    convb_non_quantized = np.array(bias[f'b{key}'], dtype=np.float32)
    convW_non_quantized = convW_non_quantized.reshape(
        [convW_non_quantized.shape[2], convW_non_quantized.shape[3], convW_non_quantized.shape[1],
         convW_non_quantized.shape[0]])

    convW_quantized = np.array(net_data[f'convolution_{key}'], dtype=np.float32)
    convb_quantized = np.array(net_data[f'bias_{key}'], dtype=np.float32)

    workbook = excel.Workbook(f'result\\convolution{key}.xlsx')
    worksheet = workbook.add_worksheet(f'convolution{key}')
    merge_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'})

    worksheet.merge_range('B1:K1', 'Memory Access', merge_format)
    worksheet.merge_range('M1:W1', 'Multiply', merge_format)
    worksheet.merge_range('Y1:AH1', 'Add', merge_format)

    # input_data_numpy = np.array(input_data.eval(session=sess))
    input_data_factorization = input_data_non_quantized.copy()
    input_data_factorization = input_data_factorization.reshape(
        [input_data_non_quantized.shape[1], input_data_non_quantized.shape[2], input_data_non_quantized.shape[3]])
    factorization.convolve(worksheet, input_data_factorization, convW_non_quantized, convb_non_quantized,
                           layer_num=f'convolution_{key}', mode='NON_QUANTIZED', stride=stride,
                           padding=padding)

    input_data_factorization = input_data_quantized.copy()
    input_data_factorization = input_data_factorization.reshape(
        [input_data_non_quantized.shape[1], input_data_non_quantized.shape[2], input_data_non_quantized.shape[3]])
    factorization.convolve(worksheet, input_data_factorization, convW_quantized, convb_quantized,
                           layer_num=f'convolution_{key}', mode='QUANTIZED', stride=stride,
                           padding=padding)

    workbook.close()

    kernel1 = tf.constant(convW_non_quantized, tf.float32)
    biases1 = tf.constant(convb_non_quantized, tf.float32)
    kernel2 = tf.constant(convW_quantized, tf.float32)
    biases2 = tf.constant(convb_quantized, tf.float32)

    temp1 = tf.nn.convolution(input_data_non_quantized, kernel1, padding, [1, stride, stride, 1])
    temp2 = tf.nn.relu(temp1)
    result1 = tf.nn.bias_add(temp2, biases1)

    temp1 = tf.nn.convolution(input_data_non_quantized, kernel2, padding, [1, stride, stride, 1])
    temp2 = tf.nn.relu(temp1)
    result2 = tf.nn.bias_add(temp2, biases2)

    return sess.run(result1), sess.run(result2)


print('layer1')
# layer 1 convolution
# conv1 = conv_layer(input_image, "conv1", stride=4)
conv1_non_quantized, conv1_quantized = conv_layer(input_image, None, 1, stride=4, padding='SAME')
# max-pooling layer1
print('conv1.shape', conv1_non_quantized.shape)
maxpool1_non_quantized = tf.nn.max_pool(conv1_non_quantized,
                                        ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
maxpool1_non_quantized = sess.run(maxpool1_non_quantized)

maxpool1_quantized = tf.nn.max_pool(conv1_quantized,
                                    ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
maxpool1_quantized = sess.run(maxpool1_quantized)

# maxpool1 = maxpool1.reshape([maxpool1.shape[1], maxpool1.shape[2], maxpool1.shape[3]])
#

print('pool1.shape', maxpool1_non_quantized.shape)
print('layer2')

# layer 2 convolution
# conv2 = conv_layer(maxpool1, "conv2", padding="SAME")
conv2_non_quantized, conv2_quantized = conv_layer(maxpool1_non_quantized, maxpool1_quantized, 2,
                                                  padding="SAME")  # max-pooling layer2

maxpool2_non_quantized = tf.nn.max_pool(conv2_non_quantized,
                                        ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
maxpool2_non_quantized = sess.run(maxpool2_non_quantized)

maxpool2_quantized = tf.nn.max_pool(conv2_quantized,
                                    ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
maxpool2_quantized = sess.run(maxpool2_quantized)

print('conv2.shape', conv2_non_quantized.shape)
print('maxpool2.shape', maxpool2_non_quantized.shape)
# maxpool2 = maxpool2.reshape([maxpool2.shape[1], maxpool2.shape[2], maxpool2.shape[3]])

print('pool2.shape', maxpool2_non_quantized.shape)
print('layer3')

# layer 3 convolution
# conv3 = conv_layer(maxpool2, "conv3", padding="SAME")
conv3_non_quantized, conv3_quantized = conv_layer(maxpool2_non_quantized, maxpool2_quantized, 3, padding="SAME")
print('layer4')
# layer 4 convolution
# conv4 = conv_layer(conv3, "conv4", padding="SAME")

print('conv3.shape', conv3_non_quantized.shape)
conv4_non_quantized, conv4_quantized = conv_layer(conv3_non_quantized, conv3_quantized, 4, padding="SAME")
print('layer5')
# layer 5 convolution
# conv5 = conv_layer(conv4, "conv5", padding="SAME")

print('conv4.shape', conv4_non_quantized.shape)
conv5_non_quantized, conv5_quantized = conv_layer(conv4_non_quantized, conv4_quantized, 5, padding="SAME")
# max-pooling layer5
# maxpool5_temp_non_quantized = tf.nn.max_pool(conv5_non_quantized,
#                                              ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
# maxpool5_non_quantized = sess.run(maxpool5_temp_non_quantized)
# maxpool5_non_quantized = maxpool5_non_quantized.reshape(
#     [maxpool5_non_quantized.shape[1], maxpool5_non_quantized.shape[2], maxpool5_non_quantized.shape[3]])
#
# print('conv5.shape', conv5_non_quantized.shape)
# print('pool5.shape', maxpool5_non_quantized.shape)
# # layer 6 fc
# print('layer6')
# fc6 = maxpool5_non_quantized.reshape(-1)
#
# # layer 7 fc
# print('layer7')
# # fc6W = np.array(net_data["fc6"][0])
# # fc6b = np.array(net_data["fc6"][1])
# fc6W = np.array(net_data["fc_1"].T)
# fc6b = np.array(net_data["fc_1_bias"])
# fc7 = fc6.dot(fc6W) + fc6b
#
# # layer 8 fc
# print('layer8')
# # fc7W = np.array(net_data["fc7"][0])
# # fc7b = np.array(net_data["fc7"][1])
# fc7W = np.array(net_data["fc_2"].T)
# fc7b = np.array(net_data["fc_2_bias"])
# fc8 = fc7.dot(fc7W) + fc7b
#
# # output
# print('output')
# fc8W = np.array(net_data["fc_3"].T)
# fc8b = np.array(net_data["fc_3_bias"])
# output = fc8.dot(fc8W) + fc8b
