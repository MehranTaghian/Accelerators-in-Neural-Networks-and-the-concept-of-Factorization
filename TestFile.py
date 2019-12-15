import numpy as np

# a = tf.constant(4, dtype=tf.float32)
# b = tf.constant(3.0)
# total = a + b
#
# print(sess.run(total))
#
# x = tf.placeholder(tf.float32)
# y = tf.placeholder(tf.float32)
# total = x + y
#
# print(sess.run(total, feed_dict={x:4, y:3}))


#######################################################################
# my_data = [
#     [0, 1],
#     [2, 3],
#     [4, 5],
#     [6, 7],
# ]
#
# slices = tf.data.Dataset.from_tensor_slices(my_data)
# next_item = slices.make_one_shot_iterator().get_next()


# r = tf.random_normal([10,3])
# dataset = tf.data.Dataset.from_tensor_slices(r)
# iterator = dataset.make_initializable_iterator()
# next_row = iterator.get_next()
#
# sess.run(iterator.initializer)
# while True:
#   try:
#     print(sess.run(next_row))
#   except tf.errors.OutOfRangeError:
#     break


# x = tf.placeholder(tf.float32, shape=[None, 3])
# linear_model = tf.layers.Dense(units=2)
# y = linear_model(x)
#
# init = tf.global_variables_initializer()
# sess.run(init)
#
# print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))

# ---------------------------------------------------------------

# features = {
# #     'sales': [[5], [10], [8], [9]],
# #     'department': ['sports', 'sports', 'gardening', 'gardening']}
# #
# # department_column = tf.feature_column.categorical_column_with_vocabulary_list(
# #     'department', ['sports', 'gardening', 'football'])
# # department_column = tf.feature_column.indicator_column(department_column)
# #
# # columns = [
# #     tf.feature_column.numeric_column('sales'),
# #     department_column
# # ]
# #
# # inputs = tf.feature_column.input_layer(features, columns)
# #
# # var_init = tf.global_variables_initializer()
# # table_init = tf.tables_initializer()
# # sess.run((var_init, table_init))
# # print(sess.run(inputs))

# ------------------------------------------------------------------

# a = tf.constant([1, 2, 3], tf.float32)
# b = tf.Print(a, [a])
# # c = b + 1
# with sess.as_default():
#     b.eval()

# ------------------------------------------------------------------

# v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
# assignment = v.assign_add(1)
# with sess.as_default():
#     tf.global_variables_initializer().run()
#     print(sess.run(assignment))  # or assignment.op.run(), or assignment.eval()


# --------------------------------------------------------------------
# v = tf.get_variable('v', [1, 2, 3])
# b = tf.Print(v, [v])
# with sess.as_default():
#     sess.run(b)

# --------------------------------------------------------------------

# import numpy as np
# import tensorflow as tf
#
# sess = tf.Session()
#
# net_data = np.load(open(r"C:\Users\Mehran\Desktop\Lotfi-Kamran\Weights\bvlc_alexnet.npy", "rb"), encoding="latin1",
#                    allow_pickle=True).item()
#
# conv1 = np.array(net_data['conv1'][0])
# v = tf.get_variable("v", shape=conv1.shape, initializer=tf.zeros_initializer())
# assignment = v.assign_add(net_data['conv1'][0])
# with sess.as_default():
#     tf.global_variables_initializer().run()
#     sess.run(assignment)  # or assignment.op.run(), or assignment.eval()
#     print(sess.run(tf.shape(v)))
# conv2 = np.array(net_data['conv2'][1])
# conv3 = np.array(net_data['conv3'])
# conv4 = np.array(net_data['conv4'])
# conv5 = np.array(net_data['conv5'])
#
# # print(conv1.shape)
#
# fc1 = np.array(net_data['fc6'])
# fc2 = np.array(net_data['fc7'][0])
# fc3 = np.array(net_data['fc8'])

# --------------------------------------------------------------------

# sess = tf.Session()
# a = tf.random_normal([3, 3], stddev=10)
# inp = tf.constant(sess.run(a))
# inp = tf.constant(1.2)
# min_range = tf.reduce_min(inp)
# max_range = tf.reduce_max(inp)
#
# result = tf.quantize(inp, -1, 4, T=tf.quint8)


# print(range())


# result = tf.quantization.quantize(
#     input=inp,
#     min_range=-5,
#     max_range=5,
#     T=tf.quint16,
#     mode='SCALED',
#     round_mode='HALF_AWAY_FROM_ZERO',
#     name=None
# )
# b = tf.Print(result, [result])

# sess.run(b)


# Increase the cell size of the merged cells to highlight the formatting.
# worksheet.set_column('B:D', 12)
# worksheet.set_row(3, 30)
# worksheet.set_row(6, 30)
# worksheet.set_row(7, 30)
#
#
# # Create a format to use in the merged range.
# merge_format = workbook.add_format({
#     'bold': 1,
#     'border': 1,
#     'align': 'center',
#     'valign': 'vcenter'})
# # 'fg_color': 'yellow'})
#
#
# # Merge 3 cells.
# worksheet.merge_range('B4:D4', 'Merged Range',merge_format)
#
# # Merge 3 cells over two rows.
# worksheet.merge_range('B7:D8', 'Merged Range', merge_format)
#
# workbook.close()

net_data = np.load(
    open(r"C:\Users\Mehran\Desktop\Desktop files\Lotfi-Kamran\Weights\INQ_AlexNet_quantized_weights_0.6753.pickle",
         "rb"),
    encoding="latin1",
    allow_pickle=True)

print(net_data.keys())

print(list(net_data.values())[0].shape)
