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

"""CIFAR example using input pipelines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('cifar_train_data_file', 'gs://ptosis-test/data/img/123689_64.jpg',
                    'Path to CIFAR10 training data.')


# def parser(serialized_example):
#   """Parses a single tf.Example into image and label tensors."""
#   features = tf.parse_single_example(
#       serialized_example,
#       features={
#           'image': tf.FixedLenFeature([], tf.string),
#           'label': tf.FixedLenFeature([], tf.int64),
#       })
#   image = tf.decode_raw(features['image'], tf.uint8)
#   image.set_shape([3*32*32])
#   # Normalize the values of the image from the range [0, 255] to [-1.0, 1.0]
#   image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
#   image = tf.transpose(tf.reshape(image, [3, 32*32]))
#   label = tf.cast(features['label'], tf.int32)
#   return image, label

def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [64, 64])
  print("L72", image_resized)
  return image_resized

# class InputFunction(object):
#   """Wrapper class that is passed as callable to Estimator."""

#   def __init__(self, is_training, noise_dim):
#     self.is_training = is_training
#     self.noise_dim = noise_dim
#     self.data_file = (FLAGS.cifar_train_data_file if is_training
#                       else FLAGS.cifar_test_data_file)

#   def __call__(self, params):
#     batch_size = params['batch_size']
#     dataset = tf.data.TFRecordDataset([self.data_file])
#     dataset = dataset.map(parser, num_parallel_calls=batch_size)
#     dataset = dataset.prefetch(4 * batch_size).cache().repeat()
#     dataset = dataset.apply(
#         tf.data.Dataset.batch(batch_size))
#     dataset = dataset.prefetch(2)
#     images, labels = dataset.make_one_shot_iterator().get_next()

#     # Reshape to give inputs statically known shapes.
#     images = tf.reshape(images, [batch_size, 32, 32, 3])

#     random_noise = tf.random_normal([batch_size, self.noise_dim])

#     features = {
#         'real_images': images,
#         'random_noise': random_noise}

#     return features, labels


class InputFunction(object):
  """Wrapper class that is passed as callable to Estimator."""

  def __init__(self, is_training, noise_dim):
    self.is_training = is_training
    self.noise_dim = noise_dim
    self.data_file = (FLAGS.cifar_train_data_file if is_training else print("Not training"))

  def __call__(self, params):
      # Batch size
      batch_size = params['batch_size']
      # A vector of filenames.
      # filenames = tf.constant(["/data/223680_64.jpg", "/data/223681_64.jpg"])
      filenames = tf.constant([self.data_file])

      dataset = tf.data.Dataset.from_tensor_slices((filenames))
      dataset = dataset.map(_parse_function)
      dataset = dataset.prefetch(4 * batch_size).cache().repeat()
      dataset = dataset.apply(batch(batch_size))
      dataset = dataset.prefetch(2)
      images = dataset.make_one_shot_iterator().get_next()
      print("L89",images)
      print("L90",images.shape)
      # Reshape to give inputs statically known shapes.
      images = tf.reshape(images, [batch_size, 64, 64, 3])
      random_noise = tf.random_normal([batch_size, self.noise_dim])

      features = {
          'real_images': images,
          'random_noise': random_noise}

      # return features
      return features


  # def __call__(self, params):
  #   # A vector of filenames.
  #   batch_size = params['batch_size']
  #   print("HELLO")
  #   filenames = tf.constant(['./data/img/223680_64.jpg'])

  #   dataset = tf.data.Dataset.from_tensor_slices((filenames))
  #   dataset = dataset.map(_parse_function, num_parallel_calls=batch_size)
  #   dataset = dataset.prefetch(4 * batch_size).cache().repeat()
  #   dataset = dataset.apply(
  #       tf.data.Dataset.batch(batch_size))
  #   dataset = dataset.prefetch(2)
  #   images = dataset.make_one_shot_iterator().get_next()

  #   # Reshape to give inputs statically known shapes.
  #   images = tf.reshape(images, [batch_size, 64, 64, 3])

  #   random_noise = tf.random_normal([batch_size, self.noise_dim])

  #   features = {
  #       'real_images': images,
  #       'random_noise': random_noise}

  #   # return features, labels
  #   return features


def convert_array_to_image(array):
  """Converts a numpy array to a PIL Image and undoes any rescaling."""
  img = Image.fromarray(np.uint8((array + 1.0) / 2.0 * 255), mode='RGB')
  return img
