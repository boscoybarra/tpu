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
"""Simple generator and discriminator models.

Based on the convolutional and "deconvolutional" models presented in
"Unsupervised Representation Learning with Deep Convolutional Generative
Adversarial Networks" by A. Radford et. al.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _leaky_relu(x):
  return tf.nn.leaky_relu(x, alpha=0.2)


def _batch_norm(x, is_training, name):
  return tf.layers.batch_normalization(
      x, momentum=0.9, epsilon=1e-5, training=is_training, name=name)


def _dense(x, channels, name):
  return tf.layers.dense(
      x, channels,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)


def _conv2d(x, filters, kernel_size, stride, name):
  return tf.layers.conv2d(
      x, filters, [kernel_size, kernel_size],
      strides=[stride, stride], padding='same',
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)


def _deconv2d(x, filters, kernel_size, stride, name):
  return tf.layers.conv2d_transpose(
      x, filters, [kernel_size, kernel_size],
      strides=[stride, stride], padding='same',
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)


def discriminator(x, is_training=True, scope='Discriminator'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    print("HEELLOO?",x)
    print("HEELLOO 2?",x.shape)
    x = _conv2d(x, 64, 5, 2, name='d_conv1')
    x = _leaky_relu(x)

    x = _conv2d(x, 128, 5, 2, name='d_conv2')
    x = _leaky_relu(_batch_norm(x, is_training, name='d_bn2'))

    x = _conv2d(x, 256, 5, 2, name='d_conv3')
    x = _leaky_relu(_batch_norm(x, is_training, name='d_bn3'))

    x = tf.reshape(x, [-1, 4 * 4 * 256])
    print("HEELLOO 74?",x)
    print("HEELLOO 76?",x.shape)

    x = _dense(x, 1, name='d_fc_4')
    print("HEELLOO 79?",x)
    print("HEELLOO 80?",x.shape)

    return x


def generator(x, is_training=True, scope='Generator'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    print("HEELLOO 87",x)
    print("HEELLOO 88?",x.shape)
    x = _dense(x, 4096, name='g_fc1')
    print("HEELLOO 90",x)
    print("HEELLOO 91?",x.shape)
    x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn1'))

    print("HEELLOO 94",x)
    print("HEELLOO 95?",x.shape)
    x = tf.reshape(x, [-1, 4, 4, 256])

    x = _deconv2d(x, 128, 5, 2, name='g_dconv2')
    x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn2'))

    x = _deconv2d(x, 64, 4, 2, name='g_dconv3')
    x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn3'))

    x = _deconv2d(x, 3, 4, 2, name='g_dconv4')
    x = tf.tanh(x)
    print("HEELLOO 106",x)
    print("HEELLOO 107?",x.shape)

    return x

