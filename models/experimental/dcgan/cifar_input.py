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
import resnet_preprocessing

FLAGS = flags.FLAGS

flags.DEFINE_string('cifar_train_data_file', '',
                    'Path to CIFAR10 training data.')
flags.DEFINE_string('cifar_test_data_file', '', 'Path to CIFAR10 test data.')


def dataset_parser(serialized_example):
    """Parses an image and its label from a serialized ResNet-50 TFExample.
    Args:
      value: serialized string containing an ImageNet TFExample.
    Returns:
      Returns a tuple of (image, label) from the TFExample.
    """
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'image/format': tf.FixedLenFeature((), tf.string, 'jpg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
        'image/class/text': tf.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

    image = self.image_preprocessing_fn(
        image_bytes=image_bytes,
        is_training=self.is_training)

    # Subtract one so that labels are in [0, 1000).
    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32) - 1

    return image, label

def parser(serialized_example):
  """Parses a single tf.Example into image and label tensors."""
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })
  image = tf.decode_raw(features['image'], tf.uint8)
  image.set_shape([3*32*32])
  # Normalize the values of the image from the range [0, 255] to [-1.0, 1.0]
  image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
  image = tf.transpose(tf.reshape(image, [3, 32*32]))
  label = tf.cast(features['label'], tf.int32)
  return image, label


class InputFunction(object):
  """Wrapper class that is passed as callable to Estimator."""

  def __init__(self, is_training, noise_dim):
    self.is_training = is_training
    self.noise_dim = noise_dim
    self.data_file = (FLAGS.cifar_train_data_file if is_training
                      else FLAGS.cifar_test_data_file)
    self.image_preprocessing_fn = resnet_preprocessing.preprocess_image

  def __call__(self, params):
    batch_size = params['batch_size']
    if not self.data_dir:
      tf.logging.info('Undefined data_dir implies null input')
      return tf.data.Dataset.range(1).repeat().map(self._get_null_input)

    # Shuffle the filenames to ensure better randomization.
    file_pattern = os.path.join(
        self.data_dir, 'gs://ptosis-test/tpu/dcgan/data/train-*' if self.is_training else 'gs://ptosis-test/tpu/dcgan/data/validation-*')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=self.is_training)

    # Shard the data into `num_replicas` parts, get the part for `replica`
    if self.num_replicas:
      dataset = dataset.shard(self.num_replicas, self.replica)

    if self.is_training and not self.cache:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=self.num_parallel_calls, sloppy=True))

    if self.cache:
      dataset = dataset.cache().apply(
          tf.contrib.data.shuffle_and_repeat(1024 * 16))
    else:
      dataset = dataset.shuffle(1024)

    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            self.dataset_parser, batch_size=batch_size,
            num_parallel_batches=self.num_cores, drop_remainder=True))


    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    images, labels = dataset.make_one_shot_iterator().get_next()

    # Reshape to give inputs statically known shapes.
    images = tf.reshape(images, [batch_size, 32, 32, 3])

    random_noise = tf.random_normal([batch_size, self.noise_dim])

    features = {
        'real_images': images,
        'random_noise': random_noise}

    return features, labels


def convert_array_to_image(array):
  """Converts a numpy array to a PIL Image and undoes any rescaling."""
  img = Image.fromarray(np.uint8((array + 1.0) / 2.0 * 255), mode='RGB')
  return img