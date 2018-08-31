import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


#定义常量
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER_NODE = 500

#设置权重
def get_weight(shape, regularizer):
	w = tf.Variable(tf.truncated_normal(shape), dtype=tf.float32)
	if regularizer != None:
		tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

#设置偏置项
def get_bias(shape):
	b = tf.Variable(tf.zeros(shape))
	return b

#定义神经网络结构
def forward(x, regularizer):

	w1 = get_weight([INPUT_NODE,LAYER_NODE], regularizer)
	b1 = get_bias([LAYER_NODE])
	y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

	w2 = get_weight([LAYER_NODE,OUTPUT_NODE], regularizer)
	b2 = get_bias([OUTPUT_NODE])
	y = tf.matmul(y1, w2) + b2

	return y


