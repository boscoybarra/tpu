import tensorflow as tf
from PIL import Image
import numpy as np
import os
from scipy.sparse import csr_matrix
import glob


# images = []
# labels = []


# dirs = os.listdir('/Users/jb/Documents/dl/tpu/models/experimental/dcgan/data/pics/')

# for d in dirs[0:669]:
#     # im = ('/Users/jb/Documents/dl/tpu/models/experimental/dcgan/data/pics/{}'.format(d))
#     im = ('gs://ptosis-test/data/{}'.format(d))
#     images.append(im)
#     print(len(images))
    
# print(images)

# for l in range(669):
#     l += 0
#     labels.append(l)
#     print(labels)
#     print(len(labels))

# convert_to(images, labels, 'output')


### The process is as follows:

# Data -> FeatureSet -> Example -> Serialized Example -> tfRecord.



# Converting the values into features
# _int64 is used for numeric values

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# _bytes is used for string/char values

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

tfrecord_filename = '/Users/jb/Desktop/output.tfrecords'

# Initiating the writer and creating the tfrecords file.

writer = tf.python_io.TFRecordWriter(tfrecord_filename)

# Loading the location of all files - image dataset
# Considering our image dataset has apple or orange
# The images are named as apple01.jpg, apple02.jpg .. , orange01.jpg .. etc.

images = glob.glob('/Users/jb/Desktop/pics/*.jpg')
for image in images[:1]:
  img = Image.open(image)
  img = np.array(img.resize((32,32)))
label = 0 if '64' in image else 1
feature = { 'label': _int64_feature(label),
              'image': _bytes_feature(img.tostring()) }

# Create an example protocol buffer

example = tf.train.Example(features=tf.train.Features(feature=feature))

# Writing the serialized example.

writer.write(example.SerializeToString())

writer.close()


### So to read it back, the process is reversed.

# tfRecord -> SerializedExample -> Example -> FeatureSet -> Data


# reader = tf.TFRecordReader()
# filenames = glob.glob('*.tfrecords')
# filename_queue = tf.train.string_input_producer(
#    filenames)
# _, serialized_example = reader.read(filename_queue)
# feature_set = { 'image': tf.FixedLenFeature([], tf.string),
#                'label': tf.FixedLenFeature([], tf.int64)
#            }
           
# features = tf.parse_single_example( serialized_example, features= feature_set )
# label = features['label']
 
# with tf.Session() as sess:
#   print sess.run([image,label])

