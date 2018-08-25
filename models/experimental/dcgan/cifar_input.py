# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Read RESNET data as TFRecords and create a tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf
import glob2

FLAGS = flags.FLAGS

flags.DEFINE_string('cifar_train_data_file', 'gs://ptosis-test/data/output.tfrecords', 'Training .tfrecord data file')
flags.DEFINE_string('cifar_test_data_file', 'gs://ptosis-test/data/output.tfrecords', 'Test .tfrecord data file')

NUM_TRAIN_IMAGES = 669
NUM_EVAL_IMAGES = 669


# def parser(serialized_example):
#   """Parses a single Example into image and label tensors."""
#   features = tf.parse_single_example(
#       serialized_example,
#       features={
#           'image_raw': tf.FixedLenFeature([], tf.string),
#           'label': tf.FixedLenFeature([], tf.int64)   # label is unused
#       })
#   image = tf.decode_raw(features['image_raw'], tf.uint8)
#   image.set_shape([3 * 64 * 64])
#   image = tf.reshape(image, [64, 64, 3])

#   # Normalize the values of the image from [0, 255] to [-1.0, 1.0]
#   image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0

#   # image = tf.transpose(tf.reshape(image, [3, 32*32]))

#   label = tf.cast(tf.reshape(features['label'], shape=[]), dtype=tf.int32)
#   return image, label

def parser(serialized_example):
  """Parses a single Example into image and label tensors."""
  reader = tf.TFRecordReader()
  filenames = glob2.glob('/home/jb/tpu/output.tfrecords')
  # filenames = "/home/jb/tpu/output.tfrecords"
    dataset = tf.data.TFRecordDataset(filenames)
  filename_queue = tf.train.string_input_producer(
     filenames)
  _, serialized_example = reader.read(filename_queue)
  feature_set = { 'image': tf.FixedLenFeature([], tf.string),
                 'label': tf.FixedLenFeature([], tf.int64)
             }
             
  features = tf.parse_single_example( serialized_example, features= feature_set )
  label = features['label']
  image = features['image']
  print("L71",image)
   
  # with tf.Session() as sess:
  #   print(sess.run([image,label]))

  image.set_shape([3 * 64 * 64])
  image = tf.reshape(image, [64, 64, 3])

  # Normalize the values of the image from [0, 255] to [-1.0, 1.0]
  image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0

  return image, label


class InputFunction(object):
  """Wrapper class that is passed as callable to Estimator."""

  def __init__(self, is_training, noise_dim):
    self.is_training = is_training
    self.noise_dim = noise_dim
    self.data_file = (FLAGS.mnist_train_data_file if is_training
                      else FLAGS.mnist_test_data_file)

  def __call__(self, params):
    """Creates a simple Dataset pipeline."""

    batch_size = params['batch_size']
    filenames = "/home/jb/tpu/output.tfrecords"
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser).cache()
    if self.is_training:
      dataset = dataset.repeat()
    dataset = dataset.shuffle(1024)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.prefetch(2)    # Prefetch overlaps in-feed with training
    images, labels = dataset.make_one_shot_iterator().get_next()

    random_noise = tf.random_normal([batch_size, self.noise_dim])

    features = {
        'real_images': images,
        'random_noise': random_noise}

    return features, labels


def convert_array_to_image(array):
  """Converts a numpy array to a PIL Image and undoes any rescaling."""
  array = array[:, :, 0]
  img = Image.fromarray(np.uint8((array + 1.0) / 2.0 * 255), mode='L')
  return img