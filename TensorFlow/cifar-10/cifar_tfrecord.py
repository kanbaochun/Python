import tensorflow as tf
from PIL import Image
import os

#生成tfrecord文件
def write_tfrecord(tfRecordName, trainPath):
	writer = tf.python_io.TFRecordWriter(tfRecordName)
	count = 0
	for folder in os.listdir(trainPath):
		trainFolder = trainPath + '/' + folder
		labels = [0] * 10
		labels[count] = 1
		count = count + 1
		for file in os.listdir(trainFolder):
			trainFile = trainFolder + '/' + file		
			img = Image.open(trainFile)
			img_raw = img.tobytes()

			example = tf.train.Example(features=tf.train.Features(feature={
					'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
					'label':tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
					}))	
			writer.write(example.SerializeToString())
		print('folder ' + str(trainFolder) + ' is writed')
	writer.close()
	print('write tfrecord successful')

def	generate_tfrecord():
	isExists = os.path.exists('data_path')
	if not isExists:
		os.makedirs('data_path')
		print('the directory was created successfully')
	else:
		print('directory already exists')
	write_tfrecord('data_path/tfrecord_train', 'cifar-10/train')
	write_tfrecord('data_path/tfrecord_test', 'cifar-10/test') 

#读取tfrecord文件
def read_tfrecord(tfrecord_path):
	fileName_queue = tf.train.string_input_producer([tfrecord_path])
	reader = tf.TFRecordReader()
	_,Serialized_example = reader.read(fileName_queue)
	features = tf.parse_single_example(Serialized_example, 
		features={
		'label':tf.FixedLenFeature([10], tf.int64),
		'img_raw':tf.FixedLenFeature([], tf.string)
		})
	img = tf.decode_raw(features['img_raw'], tf.uint8)
	img.set_shape([3072])
	img = tf.cast(img, tf.float32)* (1./255)
	label = tf.cast(features['label'], tf.float32)
	return img, label

def get_tfrecord(num, isTrain=True):
	if isTrain:
		tfrecord_path = 'data_path/tfrecord_train'
	else:
		tfrecord_path = 'data_path/tfrecord_test'
	img, label = read_tfrecord(tfrecord_path)
	img_batch, label_batch = tf.train.shuffle_batch([img, label],
												batch_size = num,
												num_threads = 2,
												capacity = 1000,
												min_after_dequeue = 700)
	return img_batch, label_batch


#制作数据集主函数
if __name__ == '__main__':
	generate_tfrecord()
