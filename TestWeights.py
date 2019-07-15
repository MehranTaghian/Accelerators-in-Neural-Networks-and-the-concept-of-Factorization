# from keras.applications import resnet50
# model = resnet50.ResNet50(include_top=True, weights='imagenet')
# model.load_weights(r"C:\Users\Mehran\Desktop\Lotfi-Kamran\Weights\resnet50\resnet50_weights_tf_dim_ordering_tf_kernels.h5")
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


# print((model.get_weights()[0]).shape)

# model = load_model(r"C:\Users\Mehran\Desktop\Lotfi-Kamran\Weights\resnet50\resnet50_weights_tf_dim_ordering_tf_kernels.h5")


# from keras.models import load_model

# model = load_model(r"C:\Users\Mehran\Desktop\Lotfi-Kamran\Weights\bvlc_alexnet.npy")
import numpy as np

# net_data = np.load(open(r"C:\Users\Mehran\Desktop\Lotfi-Kamran\Weights\VGG\conv2_1_W.npy", "rb"), encoding="latin1")
net_data = np.load(open(r"..\Weights\VGG\conv1_1_W.npy", "rb"), encoding="latin1")

# a = np.array(net_data["conv2"][0])
# print(a.shape)
#
# b = np.array(net_data["conv2"][1])
# print(b.shape)

print(net_data.shape)

# a = np.array(net_data["fc6"][0])
# print(a.shape)

import requests

print('Beginning file download with requests')

url = 'http://i3.ytimg.com/vi/J---aiyznGQ/mqdefault.jpg'
r = requests.get(url)

with open(r'C:\Users\Mehran\Desktop\mqdefault.jpg', 'wb') as f:
    f.write(r.content)

# Retrieve HTTP meta-data
print(r.status_code)
print(r.headers['content-type'])
print(r.encoding)

# np.load("http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy", encoding="latin1")


# from __future__ import absolute_import, division, print_function, unicode_literals
#
# import tensorflow as tf
#
# tf.enable_eager_execution()
#
# # layer = tf.keras.layers.Dense(100)
# layer = tf.keras.layers.Dense(10, input_shape=(None, 5))
#
# # print(layer(tf.ones([10, 5])))
# # layer(tf.ones([10, 5]))
# # print(layer.weights)
# print(tf.test.is_gpu_available())
