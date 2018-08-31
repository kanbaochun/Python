import tensorflow as tf

#定义常量
IMAGE_SIZE = 32
NUM_CHANNELS = 3
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 64
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
FC_SIZE1 = 512
FC_SIZE2 = 512
OUTPUT_NODE = 10

#设置权重
def get_weight(shape, regularizer):
	w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
	if regularizer != None:
		tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

#设置偏置项
def get_bias(shape):
	b = tf.Variable(tf.zeros(shape))
	return b

#定义卷积层1
def conv2d(x, w):
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

#最大池化
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

#定义神经网络结构
def forward(x, train ,regularizer):

	#卷积层1操作
	CONV1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
	CONV1_b = get_bias([CONV1_KERNEL_NUM])
	conv1 = conv2d(x, CONV1_w)
	relu1 = tf.nn.relu(tf.nn.bias_add(conv1, CONV1_b))
	pool1 = max_pool_2x2(relu1)

	#卷积层2操作
	CONV2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
	CONV2_b = get_bias([CONV2_KERNEL_NUM])
	conv2 = conv2d(pool1, CONV2_w)
	relu2 = tf.nn.relu(tf.nn.bias_add(conv2, CONV2_b))
	pool2 = max_pool_2x2(relu2)

	#建立全连接网络
	pool_shape = pool2.get_shape().as_list()
	nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
	reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

	#隐藏层1计算
	fc1_w = get_weight([nodes, FC_SIZE1], regularizer)
	fc1_b = get_bias([FC_SIZE1])
	fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
	if train: fc1 = tf.nn.dropout(fc1, 0.5)

	#隐藏层1计算
	fc2_w = get_weight([FC_SIZE1, FC_SIZE2], regularizer)
	fc2_b = get_bias([FC_SIZE2])
	fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)
	if train: fc2 = tf.nn.dropout(fc2, 0.5)

	#输出层计算
	fc3_w = get_weight([FC_SIZE2, OUTPUT_NODE], regularizer)
	fc3_b = get_bias([OUTPUT_NODE])
	y = tf.matmul(fc2, fc3_w) + fc3_b

	return y


