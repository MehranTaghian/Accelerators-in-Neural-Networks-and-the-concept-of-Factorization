import numpy as np
import tensorflow as tf
from Sparsity_Aware_Accelerator import convolution
from openpyxl import Workbook

input_image = np.load('test_image.npy')
input_image = tf.constant(input_image, dtype=tf.float32)
input_image = tf.reshape(input_image, [1, input_image.shape[0], input_image.shape[1], input_image.shape[2]])
sess = tf.Session()
input_image = np.array(sess.run(input_image))
# net_data = np.load(open(r"C:\Users\Mehran\Desktop\Lotfi-Kamran\Weights\bvlc_alexnet.npy", "rb"), encoding="latin1",
#                    allow_pickle=True).item()

# net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1",
#                    allow_pickle=True).item()
# net_data = np.load(open(r"C:\Users\Mehran\Desktop\Desktop files\Lotfi-Kamran\Weights\AlexNet_quantized_weights_4.pickle", "rb"),
#                    encoding="latin1",
#                    allow_pickle=True)


def weight_preprocess(weight_dict):
    new_weight_dict = {}
    val = list(weight_dict.values())
    index = 1
    for i in range(0, len(val), 2):
        print(val[i].shape)
        print(val[i + 1].shape)

        new_weight_dict[f'convolution_{index}'] = val[i]
        new_weight_dict[f'bias_{index}'] = val[i + 1]
        index += 1

    return new_weight_dict


net_data = np.load(
    open(r"C:\Users\Mehran\Desktop\Desktop files\Lotfi-Kamran\Weights\INQ_AlexNet_quantized_weights_0.6753.pickle",
         "rb"),
    encoding="latin1",
    allow_pickle=True)

net_data = weight_preprocess(net_data)

book = Workbook()
worksheet = book.active


def convert_to_8bits(input32bits):
    r_max = np.max(input32bits)
    r_min = np.min(input32bits)
    converted = np.int8(256 / (r_max - r_min) * input32bits)
    return converted


def conv_layer(input_data, key, stride=1, padding="VALID"):
    # convW = np.array(net_data[key][0])
    # convb = np.array(net_data[key][1])
    convW = np.array(net_data[f'convolution_{key}'], dtype=np.float32)
    convb = np.array(net_data[f'bias_{key}'], dtype=np.float32)
    # convW = convW.reshape([convW.shape[2], convW.shape[3], convW.shape[1], convW.shape[0]])

    # input_data_numpy = np.array(input_data.eval(session=sess))
    input_data_for_ESAA = input_data.copy()
    input_data_for_ESAA = input_data_for_ESAA.reshape(
        [input_data.shape[1], input_data.shape[2], input_data.shape[3]])

    # input_data_for_ESAA = convert_to_8bits(input_data_for_ESAA)
    input_data_ESAA = input_data_for_ESAA.astype(np.float16)

    convolution(worksheet, input_data_for_ESAA, convW, convb, layer_num=key, stride=stride, padding=padding)

    kernel = tf.constant(convW, tf.float32)
    biases = tf.constant(convb, tf.float32)
    temp1 = tf.nn.convolution(input_data, kernel, padding, [1, stride, stride, 1])
    temp2 = tf.nn.bias_add(temp1, biases)
    result = tf.nn.relu(temp2)
    return sess.run(result)


print('layer1')
# layer 1 convolution
# conv1 = conv_layer(input_image, "conv1", stride=4)
conv1 = conv_layer(input_image, 1, stride=4, padding='SAME')
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
conv2 = conv_layer(maxpool1, 2, padding="SAME")  # max-pooling layer2
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
conv4 = conv_layer(conv3, 4, padding="SAME")
print('layer5')
# layer 5 convolution
# conv5 = conv_layer(conv4, "conv5", padding="SAME")

print('conv4.shape', conv4.shape)
conv5 = conv_layer(conv4, 5, padding="SAME")
# max-pooling layer5
maxpool5_temp = tf.nn.max_pool(conv5,
                               ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
maxpool5 = sess.run(maxpool5_temp)
maxpool5 = maxpool5.reshape([maxpool5.shape[1], maxpool5.shape[2], maxpool5.shape[3]])

book.save('result\\alex_net_result_for_ESAA.xlsx')

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
