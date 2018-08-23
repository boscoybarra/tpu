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
import os

FLAGS = flags.FLAGS

flags.DEFINE_string('cifar_train_data_file', 'gs://ptosis-test/data/train-00000-of-00001',
                    'Path to CIFAR10 training data.')
flags.DEFINE_string('cifar_test_data_file', 'gs://ptosis-test/data/validation-00000-of-00001', 'Path to CIFAR10 test data.')
flags.DEFINE_string('data_dir', 'gs://ptosis-test/data/', 'Directory where input data is stored.')
flags.DEFINE_integer('initial_shuffle_buffer_size', 4096,'Number of elements from dataset that shuffler will sample from. ''This shuffling is done before any other operations. ''Set to 0 to disable')
flags.DEFINE_integer('followup_shuffle_buffer_size', 0,'Number of elements from dataset that shuffler will sample from. ''This shuffling is done after prefetching is done. ''Set to 0 to disable')
flags.DEFINE_integer('num_files_infeed', 1, 'Number of training files to read in parallel.')
flags.DEFINE_integer('prefetch_dataset_buffer_size', 1*4096*4096,'Number of bytes in read buffer. 0 means no buffering.')
flags.DEFINE_integer('num_parallel_calls', default=64, help=('Number of parallel threads in CPU for the input pipeline'))




class InputFunction(object):
  """Wrapper class that is passed as callable to Estimator."""

  def __init__(self, is_training, noise_dim, use_bfloat16, image_size=64, num_cores=1):
    self.is_training = is_training
    self.noise_dim = noise_dim
    self.data_file = (FLAGS.cifar_train_data_file if is_training
                      else FLAGS.cifar_test_data_file)
    self.num_cores = num_cores
    self.image_preprocessing_fn = resnet_preprocessing.preprocess_image
    self.image_size = image_size
    self.use_bfloat16 = use_bfloat16

  def __call__(self, params):
    # Storage
    file_pattern = os.path.join(
        FLAGS.data_dir, 'train-*' if self.is_training else 'validation-*')
    dataset = tf.data.Dataset.list_files(file_pattern)
    if self.is_training and FLAGS.initial_shuffle_buffer_size > 0:
      dataset = dataset.shuffle(
          buffer_size=FLAGS.initial_shuffle_buffer_size)
    if self.is_training:
      dataset = dataset.repeat()

    def prefetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(
          filename, buffer_size=FLAGS.prefetch_dataset_buffer_size)
      return dataset

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            prefetch_dataset,
            cycle_length=FLAGS.num_files_infeed,
            sloppy=True))
    if FLAGS.followup_shuffle_buffer_size > 0:
      dataset = dataset.shuffle(
          buffer_size=FLAGS.followup_shuffle_buffer_size)

    # Preprocessing
    batch_size = params['batch_size']
    dataset = dataset.map(
        self.parser,
        num_parallel_calls=FLAGS.num_parallel_calls)

    dataset = dataset.prefetch(batch_size)
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.prefetch(2)  # Prefetch overlaps in-feed with training
    images, labels = dataset.make_one_shot_iterator().get_next()


    # Reshape to give inputs statically known shapes.
    # images = tf.reshape(images, [batch_size, 64, 64, 3])

    random_noise = tf.random_normal([batch_size, self.noise_dim])

    features = {
        'real_images': images,
        'random_noise': random_noise}

    # Transfer
    return features, labels


  def parser(self, value):
    """Parses a single tf.Example into image and label tensors."""
    keys_to_features = {
            'image': tf.FixedLenFeature((), tf.string, ''),
            'label': tf.FixedLenFeature([], tf.int64),
        }
    print("YOOO",keys_to_features.image)

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image'], shape=[])

    image = self.image_preprocessing_fn(
        image_bytes=image_bytes,
        is_training=self.is_training,
        image_size=self.image_size,
        use_bfloat16=self.use_bfloat16)

      # Subtract one so that labels are in [0, 1000).
    label = tf.cast(
        tf.reshape(parsed['label'], shape=[]), dtype=tf.int32) - 1

    
    # image.set_shape([3*64*64])
    # Normalize the values of the image from the range [0, 255] to [-1.0, 1.0]
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
    image = tf.transpose(tf.reshape(image, [3, 64*64]))
    return image, label


def convert_array_to_image(array):
  """Converts a numpy array to a PIL Image and undoes any rescaling."""
  img = Image.fromarray(np.uint8((array + 1.0) / 2.0 * 255), mode='RGB')
  return img