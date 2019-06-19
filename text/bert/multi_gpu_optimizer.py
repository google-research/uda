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
"""Functions and classes related to optimization (weight updates).

Copied from https://github.com/JayYip/bert-multitask-learning/blob/master/bert_multitask_learning/optimizer.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import resource_variable_ops


class AdamWeightDecayOptimizer(optimizer.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(
      self,
      learning_rate,
      weight_decay_rate=0.0,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=None,
      name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def _prepare(self):
    self.learning_rate_t = ops.convert_to_tensor(
        self.learning_rate, name='learning_rate')
    self.weight_decay_rate_t = ops.convert_to_tensor(
        self.weight_decay_rate, name='weight_decay_rate')
    self.beta_1_t = ops.convert_to_tensor(self.beta_1, name='beta_1')
    self.beta_2_t = ops.convert_to_tensor(self.beta_2, name='beta_2')
    self.epsilon_t = ops.convert_to_tensor(self.epsilon, name='epsilon')

  def _create_slots(self, var_list):
    for v in var_list:
      self._zeros_slot(v, 'm', self._name)
      self._zeros_slot(v, 'v', self._name)

  def _apply_dense(self, grad, var):
    learning_rate_t = math_ops.cast(
        self.learning_rate_t, var.dtype.base_dtype)
    beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
    beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
    weight_decay_rate_t = math_ops.cast(
        self.weight_decay_rate_t, var.dtype.base_dtype)

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    # Standard Adam update.
    next_m = (
        tf.multiply(beta_1_t, m) +
        tf.multiply(1.0 - beta_1_t, grad))
    next_v = (
        tf.multiply(beta_2_t, v) + tf.multiply(1.0 - beta_2_t,
                                               tf.square(grad)))

    update = next_m / (tf.sqrt(next_v) + epsilon_t)

    if self._do_use_weight_decay(var.name):
      update += weight_decay_rate_t * var

    update_with_lr = learning_rate_t * update

    next_param = var - update_with_lr

    return control_flow_ops.group(*[var.assign(next_param),
                                    m.assign(next_m),
                                    v.assign(next_v)])

  def _resource_apply_dense(self, grad, var):
    learning_rate_t = math_ops.cast(
        self.learning_rate_t, var.dtype.base_dtype)
    beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
    beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
    weight_decay_rate_t = math_ops.cast(
        self.weight_decay_rate_t, var.dtype.base_dtype)

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    # Standard Adam update.
    next_m = (
        tf.multiply(beta_1_t, m) +
        tf.multiply(1.0 - beta_1_t, grad))
    next_v = (
        tf.multiply(beta_2_t, v) + tf.multiply(1.0 - beta_2_t,
                                               tf.square(grad)))

    update = next_m / (tf.sqrt(next_v) + epsilon_t)

    if self._do_use_weight_decay(var.name):
      update += weight_decay_rate_t * var

    update_with_lr = learning_rate_t * update

    next_param = var - update_with_lr

    return control_flow_ops.group(*[var.assign(next_param),
                                    m.assign(next_m),
                                    v.assign(next_v)])

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    learning_rate_t = math_ops.cast(
        self.learning_rate_t, var.dtype.base_dtype)
    beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
    beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
    weight_decay_rate_t = math_ops.cast(
        self.weight_decay_rate_t, var.dtype.base_dtype)

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    m_t = state_ops.assign(m, m * beta_1_t,
                           use_locking=self._use_locking)

    m_scaled_g_values = grad * (1 - beta_1_t)
    with ops.control_dependencies([m_t]):
      m_t = scatter_add(m, indices, m_scaled_g_values)

    v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
    v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = scatter_add(v, indices, v_scaled_g_values)

    update = m_t / (math_ops.sqrt(v_t) + epsilon_t)

    if self._do_use_weight_decay(var.name):
      update += weight_decay_rate_t * var

    update_with_lr = learning_rate_t * update

    var_update = state_ops.assign_sub(var,
                                      update_with_lr,
                                      use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def _apply_sparse(self, grad, var):
    return self._apply_sparse_shared(
        grad.values, var, grad.indices,
        lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
            x, i, v, use_locking=self._use_locking))

  def _resource_scatter_add(self, x, i, v):
    with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_add(
            x.handle, i, v)]):
      return x.value()

  def _resource_apply_sparse(self, grad, var, indices):
    return self._apply_sparse_shared(
        grad, var, indices, self._resource_scatter_add)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True
