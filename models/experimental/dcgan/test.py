import tensorflow as tf
from PIL import Image
import numpy as np
import os


# Get images and labels into .bin locally

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# images and labels array as input
def convert_to(images, labels, name):
  num_examples = labels.shape[0]
  if images.shape[0] != num_examples:
    raise ValueError("Images size %d does not match label size %d." %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())

images = []
labels = []


dirs = os.listdir('/Users/jb/Documents/dl/tpu/models/experimental/dcgan/data/pics/')

# How many images you want to transform
# total_img_to_load = 669

for d in dirs[0:6]:
    im = ('/Users/jb/Documents/dl/tpu/models/experimental/dcgan/data/pics/{}'.format(d))
    images.append(im)
    print(images)
    print(len(images))

for l in range(6):
    l = 0
    labels.append(l)
    print(labels)
    print(len(labels))


convert_to(images, labels, 'output')

