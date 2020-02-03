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
"""Helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def decay_weights(cost, weight_decay_rate):
  """Calculates the loss for l2 weight decay and adds it to `cost`."""
  costs = []
  for var in tf.trainable_variables():
    costs.append(tf.nn.l2_loss(var))
  cost += tf.multiply(weight_decay_rate, tf.add_n(costs))
  return cost


def get_TPU_estimator(FLAGS, model_fn, model_dir=None):
  ##### Create TPUEstimator
  # TPU Configuration
  if FLAGS.use_tpu:
    if FLAGS.tpu:
      tpu_cluster = tf.contrib.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    else:
      tpu_cluster = None
    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True)
  else:
    tpu_cluster = None
    session_config = None
  per_host_input = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster,
      master=FLAGS.master,
      model_dir=model_dir or FLAGS.model_dir,
      session_config=session_config,
      tpu_config=tf.contrib.tpu.TPUConfig(
          # if there is name for the job, then add a name here
          iterations_per_loop=FLAGS.iterations,
          # num_shards=FLAGS.num_core_per_host * FLAGS.num_hosts,
          per_host_input_for_training=per_host_input),
      keep_checkpoint_max=FLAGS.max_save,
      save_checkpoints_secs=None,
      save_checkpoints_steps=FLAGS.save_steps
  )

  # TPU Estimator
  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      params={"model_dir": model_dir or FLAGS.model_dir},
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)
  return estimator


def construct_scalar_host_call(
    metric_dict,
    model_dir,
    prefix="",
    reduce_fn=None):

  metric_names = list(metric_dict.keys())

  def host_call_fn(global_step, *args):
    step = global_step[0]
    with tf.contrib.summary.create_file_writer(
        logdir=model_dir, filename_suffix=".host_call").as_default():
      with tf.contrib.summary.always_record_summaries():
        for i, name in enumerate(metric_names):
          if reduce_fn is None:
            scalar = args[i][0]
          else:
            scalar = reduce_fn(args[i])
          with tf.contrib.summary.record_summaries_every_n_global_steps(1000, step):
            tf.contrib.summary.scalar(prefix + name, scalar, step=step)

        return tf.contrib.summary.all_summary_ops()

  global_step_tensor = tf.reshape(tf.train.get_or_create_global_step(), [1])
  other_tensors = [tf.reshape(metric_dict[key], [-1]) for key in metric_names]

  return host_call_fn, [global_step_tensor] + other_tensors


def get_all_variable():
  var_list = tf.trainable_variables() + tf.get_collection('moving_vars')
  for v in tf.global_variables():
    # We maintain ema for batch norm moving mean and variance as well.
    if 'moving_mean' in v.name or 'moving_variance' in v.name:
      var_list.append(v)
  var_list = list(set(var_list))
  var_list = sorted(var_list, key=lambda var: var.name)
  return var_list
