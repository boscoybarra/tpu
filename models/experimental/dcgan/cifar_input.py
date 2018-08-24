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

import abc
from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf
import resnet_preprocessing
import os
import functools

FLAGS = flags.FLAGS

flags.DEFINE_string('cifar_train_data_file', 'gs://ptosis-test/data/train-00000-of-00001',
                    'Path to CIFAR10 training data.')
flags.DEFINE_string('cifar_test_data_file', 'gs://ptosis-test/data/validation-00000-of-00001', 'Path to CIFAR10 test data.')
# flags.DEFINE_string('data_dir', 'gs://ptosis-test/data/', 'Directory where input data is stored.')
flags.DEFINE_integer('initial_shuffle_buffer_size', 1024,'Number of elements from dataset that shuffler will sample from. ''This shuffling is done before any other operations. ''Set to 0 to disable')
flags.DEFINE_integer('followup_shuffle_buffer_size', 0,'Number of elements from dataset that shuffler will sample from. ''This shuffling is done after prefetching is done. ''Set to 0 to disable')
flags.DEFINE_integer('num_files_infeed', 1, 'Number of training files to read in parallel.')
flags.DEFINE_integer('prefetch_dataset_buffer_size', 1*1024*1024,'Number of bytes in read buffer. 0 means no buffering.')
flags.DEFINE_integer('num_parallel_calls', default=64, help=('Number of parallel threads in CPU for the input pipeline'))


class ImageNetTFExampleInput(object):
  """Base class for ImageNet input_fn generator.
  Args:
    is_training: `bool` for whether the input is for training
    use_bfloat16: If True, use bfloat16 precision; else use float32.
    transpose_input: 'bool' for whether to use the double transpose trick
    num_cores: `int` for the number of TPU cores
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               is_training,
               use_bfloat16,
               num_cores=8,
               image_size=224,
               transpose_input=False):
    self.image_preprocessing_fn = resnet_preprocessing.preprocess_image
    self.is_training = is_training
    self.use_bfloat16 = use_bfloat16
    self.num_cores = num_cores
    self.transpose_input = transpose_input
    self.image_size = image_size

  def set_shapes(self, batch_size, images, labels):
    """Statically set the batch_size dimension."""
    if self.transpose_input:
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([None, None, None, batch_size])))
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size])))
    else:
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([batch_size, None, None, None])))
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size])))

    return images, labels

  def dataset_parser(self, value):
    """Parses an image and its label from a serialized ResNet-50 TFExample.
    Args:
      value: serialized string containing an ImageNet TFExample.
    Returns:
      Returns a tuple of (image, label) from the TFExample.
    """
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'image/format': tf.FixedLenFeature((), tf.string, 'jpeg'),
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
        is_training=self.is_training,
        image_size=self.image_size,
        use_bfloat16=self.use_bfloat16)

    # Subtract one so that labels are in [0, 1000).
    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32) - 1

    return image, label

  @abc.abstractmethod
  def make_source_dataset(self):
    """Makes dataset of serialized TFExamples.
    The returned dataset will contain `tf.string` tensors, but these strings are
    serialized `TFExample` records that will be parsed by `dataset_parser`.
    If self.is_training, the dataset should be infinite.
    Returns:
      A `tf.data.Dataset` object.
    """
    return

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.
    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.
    Returns:
      A `tf.data.Dataset` object.
    """
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # tf.contrib.tpu.RunConfig for details.
    batch_size = params['batch_size']

    dataset = self.make_source_dataset()

    # Use the fused map-and-batch operation.
    #
    # For XLA, we must used fixed shapes. Because we repeat the source training
    # dataset indefinitely, we can use `drop_remainder=True` to get fixed-size
    # batches without dropping any training examples.
    #
    # When evaluating, `drop_remainder=True` prevents accidentally evaluating
    # the same image twice by dropping the final batch if it is less than a full
    # batch size. As long as this validation is done with consistent batch size,
    # exactly the same images will be used.
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            self.dataset_parser, batch_size=batch_size,
            num_parallel_batches=self.num_cores, drop_remainder=True))

    # Transpose for performance on TPU
    if self.transpose_input:
      dataset = dataset.map(
          lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
          num_parallel_calls=self.num_cores)

    # Assign static batch size dimension
    dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset
    

class ImageNetInput(ImageNetTFExampleInput):
  """Generates ImageNet input_fn from a series of TFRecord files.
  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:
      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024
  The validation data is in the same format but sharded in 128 files.
  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py
  """

  def __init__(self,
               is_training,
               noise_dim,
               use_bfloat16,
               transpose_input,
               data_dir,
               num_parallel_calls=64,
               cache=False,
               num_replicas=None,
               replica=0):
    """Create an input from TFRecord files.
    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      transpose_input: 'bool' for whether to use the double transpose trick
      data_dir: `str` for the directory of the training and validation data;
          if 'null' (the literal string 'null') or implicitly False
          then construct a null pipeline, consisting of empty images
          and blank labels.
      num_parallel_calls: concurrency level to use when reading data from disk.
      cache: if true, fill the dataset by repeating from its cache
      num_replicas: `int` for the number of model replicas this dataset should
          be sharded onto, or `None` if this dataset should not be sharded.
      replica: `int` for the replica that input_fn should produce data for
    """
    super(ImageNetInput, self).__init__(
        is_training=is_training,
        noise_dim = noise_dim,
        use_bfloat16=use_bfloat16,
        transpose_input=transpose_input)
    self.data_dir = data_dir
    # TODO(b/112427086):  simplify the choice of input source
    if self.data_dir == 'null' or not self.data_dir:
      self.data_dir = None
    self.num_parallel_calls = num_parallel_calls
    self.cache = cache
    self.num_replicas = num_replicas
    self.replica = replica
    self.noise_dim

  def _get_null_input(self, data):
    """Returns a null image (all black pixels).
    Args:
      data: element of a dataset, ignored in this method, since it produces
          the same null image regardless of the element.
    Returns:
      a tensor representing a null image.
    """
    del data  # Unused since output is constant regardless of input
    return tf.zeros([224, 224, 3], tf.bfloat16
                    if self.use_bfloat16 else tf.float32)

  def dataset_parser(self, value):
    """See base class."""
    if not self.data_dir:
      return value, tf.constant(0, tf.int32)
    return super(ImageNetInput, self).dataset_parser(value)

  def make_source_dataset(self):
    """See base class."""
    if not self.data_dir:
      tf.logging.info('Undefined data_dir implies null input')
      return tf.data.Dataset.range(1).repeat().map(self._get_null_input)

    # Shuffle the filenames to ensure better randomization.
    file_pattern = os.path.join(
        self.data_dir, 'train-*' if self.is_training else 'validation-*')
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

    images, labels = dataset.make_one_shot_iterator().get_next()


    # Reshape to give inputs statically known shapes.
    images = tf.reshape(images, [batch_size, 64, 64, 3])

    random_noise = tf.random_normal([batch_size, self.noise_dim])

    features = {
        'real_images': images,
        'random_noise': random_noise}

    # Transfer
    return features, labels
    





def convert_array_to_image(array):
  """Converts a numpy array to a PIL Image and undoes any rescaling."""
  img = Image.fromarray(np.uint8((array + 1.0) / 2.0 * 255), mode='RGB')
  return img