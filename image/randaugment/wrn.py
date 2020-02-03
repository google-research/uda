# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
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
"""Builds the WideResNet Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from randaugment import custom_ops as ops

import numpy as np
import tensorflow as tf



def residual_block(
    x, in_filter, out_filter, stride, update_bn=True):
  """Adds residual connection to `x` in addition to applying BN->ReLU->3x3 Conv.

  Args:
    x: Tensor that is the output of the previous layer in the model.
    in_filter: Number of filters `x` has.
    out_filter: Number of filters that the output of this layer will have.
    stride: Integer that specified what stride should be applied to `x`.

  Returns:
    A Tensor that is the result of applying two sequences of BN->ReLU->3x3 Conv
    and then adding that Tensor to `x`.
  """

  orig_x = x
  block_x = x
  with tf.variable_scope('residual_only_activation'):
    block_x = ops.batch_norm(block_x, update_stats=update_bn,
                              scope='init_bn')
    block_x = tf.nn.relu(block_x)

  with tf.variable_scope('sub1'):
    block_x = ops.conv2d(
        block_x, out_filter, 3, stride=stride, scope='conv1')

  with tf.variable_scope('sub2'):
    block_x = ops.batch_norm(block_x, update_stats=update_bn, scope='bn2')
    block_x = tf.nn.relu(block_x)
    block_x = ops.conv2d(
        block_x, out_filter, 3, stride=1, scope='conv2')

  if stride != 1 or out_filter != in_filter:
    orig_x = ops.conv2d(
        orig_x, out_filter, 1, stride=stride, scope='conv3')
  x = orig_x + block_x
  return x


def build_wrn_model(images, num_classes, wrn_size, update_bn=True):
  """Builds the WRN model.

  Build the Wide ResNet model from https://arxiv.org/abs/1605.07146.

  Args:
    images: Tensor of images that will be fed into the Wide ResNet Model.
    num_classes: Number of classed that the model needs to predict.
    wrn_size: Parameter that scales the number of filters in the Wide ResNet
      model.

  Returns:
    The logits of the Wide ResNet model.
  """
  # wrn_size = 16 * widening factor k
  kernel_size = wrn_size
  filter_size = 3
  # depth = num_blocks_per_resnet * 6 + 4 = 28
  num_blocks_per_resnet = 4
  filters = [
      min(kernel_size, 16), kernel_size, kernel_size * 2, kernel_size * 4
  ]
  strides = [1, 2, 2]  # stride for each resblock

  # Run the first conv
  with tf.variable_scope('init'):
    x = images
    output_filters = filters[0]
    x = ops.conv2d(x, output_filters, filter_size, scope='init_conv')

  first_x = x  # Res from the beginning
  orig_x = x  # Res from previous block

  for block_num in range(1, 4):
    with tf.variable_scope('unit_{}_0'.format(block_num)):
      x = residual_block(
          x,
          filters[block_num - 1],
          filters[block_num],
          strides[block_num - 1],
          update_bn=update_bn)
    for i in range(1, num_blocks_per_resnet):
      with tf.variable_scope('unit_{}_{}'.format(block_num, i)):
        x = residual_block(
            x,
            filters[block_num],
            filters[block_num],
            1,
            update_bn=update_bn)
  with tf.variable_scope('unit_last'):
    x = ops.batch_norm(x, scope='final_bn')
    x = tf.nn.relu(x)
    x = ops.global_avg_pool(x)
    logits = ops.fc(x, num_classes)
  return logits
