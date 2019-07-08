import numpy as np
import tensorflow as tf

sess = tf.Session()

net_data = np.load(open(r"C:\Users\Mehran\Desktop\Lotfi-Kamran\Weights\bvlc_alexnet.npy", "rb"), encoding="latin1",
                   allow_pickle=True).item()

conv1 = np.array(net_data['conv1'][0])
v = tf.get_variable("v", shape=conv1.shape, initializer=tf.zeros_initializer())
assignment = v.assign_add(net_data['conv1'][0])
with sess.as_default():
    tf.global_variables_initializer().run()
    sess.run(assignment)  # or assignment.op.run(), or assignment.eval()
    print(sess.run(tf.shape(v[0,0])))


conv2 = np.array(net_data['conv2'][1])
conv3 = np.array(net_data['conv3'])
conv4 = np.array(net_data['conv4'])
conv5 = np.array(net_data['conv5'])

# print(conv1.shape)

fc1 = np.array(net_data['fc6'])
fc2 = np.array(net_data['fc7'][0])
fc3 = np.array(net_data['fc8'])
