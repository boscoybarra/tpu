import tensorflow as tf 
import glob
reader = tf.TFRecordReader()
filenames = glob.glob('/home/jb/tpu/output.tfrecords')
filename_queue = tf.train.string_input_producer(
   filenames)
_, serialized_example = reader.read(filename_queue)
feature_set = { 'image': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.int64)
           }
           
features = tf.parse_single_example( serialized_example, features= feature_set )
label = features['label']
 
with tf.Session() as sess:
  print sess.run([image,label])