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
"""Code for using the labeled examples and unlabeled examples in unsupervised data augmentation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import tensorflow as tf

from bert import modeling
from bert import optimization
from utils import tpu_utils

flags = tf.flags
FLAGS = flags.FLAGS


def kl_for_log_probs(log_p, log_q):
  p = tf.exp(log_p)
  neg_ent = tf.reduce_sum(p * log_p, axis=-1)
  neg_cross_ent = tf.reduce_sum(p * log_q, axis=-1)
  kl = neg_ent - neg_cross_ent
  return kl


def hidden_to_logits(hidden, is_training, num_classes, scope):
  hidden_size = hidden.shape[-1].value

  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    output_weights = tf.get_variable(
        "output_weights", [num_classes, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_classes], initializer=tf.zeros_initializer())

    if is_training:
      # I.e., 0.1 dropout
      hidden = tf.nn.dropout(hidden, keep_prob=0.9)

    if hidden.shape.ndims == 3:
      logits = tf.einsum("bid,nd->bin", hidden, output_weights)
    else:
      logits = tf.einsum("bd,nd->bn", hidden, output_weights)
    logits = tf.nn.bias_add(logits, output_bias)

  return logits


def get_tsa_threshold(schedule, global_step, num_train_steps, start, end):
  training_progress = tf.to_float(global_step) / tf.to_float(num_train_steps)
  if schedule == "linear_schedule":
    threshold = training_progress
  elif schedule == "exp_schedule":
    scale = 5
    threshold = tf.exp((training_progress - 1) * scale)
    # [exp(-5), exp(0)] = [1e-2, 1]
  elif schedule == "log_schedule":
    scale = 5
    # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
    threshold = 1 - tf.exp((-training_progress) * scale)
  return threshold * (end - start) + start


def create_model(
    bert_config,
    is_training,
    input_ids,
    input_mask,
    input_type_ids,
    labels,
    num_labels,
    use_one_hot_embeddings,
    tsa,
    unsup_ratio,
    global_step,
    num_train_steps,
    ):

  num_sample = input_ids.shape[0].value
  if is_training:
    assert num_sample % (1 + 2 * unsup_ratio) == 0
    sup_batch_size = num_sample // (1 + 2 * unsup_ratio)
    unsup_batch_size = sup_batch_size * unsup_ratio
  else:
    sup_batch_size = num_sample
    unsup_batch_size = 0

  pooled = modeling.bert_model(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=input_type_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  clas_logits = hidden_to_logits(
      hidden=pooled,
      is_training=is_training,
      num_classes=num_labels,
      scope="classifier")

  log_probs = tf.nn.log_softmax(clas_logits, axis=-1)
  correct_label_probs = None

  with tf.variable_scope("sup_loss"):
    sup_log_probs = log_probs[:sup_batch_size]
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    tgt_label_prob = one_hot_labels

    per_example_loss = -tf.reduce_sum(tgt_label_prob * sup_log_probs, axis=-1)
    loss_mask = tf.ones_like(per_example_loss, dtype=per_example_loss.dtype)
    correct_label_probs = tf.reduce_sum(
        one_hot_labels * tf.exp(sup_log_probs), axis=-1)

    if tsa:
      tsa_start = 1. / num_labels
      tsa_threshold = get_tsa_threshold(
          tsa, global_step, num_train_steps,
          tsa_start, end=1)

      larger_than_threshold = tf.greater(
          correct_label_probs, tsa_threshold)
      loss_mask = loss_mask * (1 - tf.cast(larger_than_threshold, tf.float32))
    else:
      tsa_threshold = 1

    loss_mask = tf.stop_gradient(loss_mask)
    per_example_loss = per_example_loss * loss_mask
    sup_loss = (tf.reduce_sum(per_example_loss) /
                tf.maximum(tf.reduce_sum(loss_mask), 1))

  unsup_loss_mask = None
  if is_training and unsup_ratio > 0:
    with tf.variable_scope("unsup_loss"):
      ori_start = sup_batch_size
      ori_end = ori_start + unsup_batch_size
      aug_start = sup_batch_size + unsup_batch_size
      aug_end = aug_start + unsup_batch_size

      ori_log_probs = log_probs[ori_start : ori_end]
      aug_log_probs = log_probs[aug_start : aug_end]
      unsup_loss_mask = 1
      if FLAGS.uda_softmax_temp != -1:
        tgt_ori_log_probs = tf.nn.log_softmax(
            clas_logits[ori_start : ori_end] / FLAGS.uda_softmax_temp,
            axis=-1)
        tgt_ori_log_probs = tf.stop_gradient(tgt_ori_log_probs)
      else:
        tgt_ori_log_probs = tf.stop_gradient(ori_log_probs)

      if FLAGS.uda_confidence_thresh != -1:
        largest_prob = tf.reduce_max(tf.exp(ori_log_probs), axis=-1)
        unsup_loss_mask = tf.cast(tf.greater(
            largest_prob, FLAGS.uda_confidence_thresh), tf.float32)
        unsup_loss_mask = tf.stop_gradient(unsup_loss_mask)

      per_example_kl_loss = kl_for_log_probs(
          tgt_ori_log_probs, aug_log_probs) * unsup_loss_mask
      unsup_loss = tf.reduce_mean(per_example_kl_loss)

  else:
    unsup_loss = 0.

  return (sup_loss, unsup_loss, clas_logits[:sup_batch_size],
          per_example_loss, loss_mask,
          tsa_threshold, unsup_loss_mask, correct_label_probs)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name_to_variable[name]
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def model_fn_builder(
    bert_config,
    init_checkpoint,
    learning_rate,
    clip_norm,
    num_train_steps,
    num_warmup_steps,
    use_tpu,
    use_one_hot_embeddings,
    num_labels,
    unsup_ratio,
    uda_coeff,
    tsa,
    print_feature=True,
    print_structure=True):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""
    if print_feature:
      tf.logging.info("*** Features ***")
      for name in sorted(features.keys()):
        tf.logging.info(
            "  name = %s, shape = %s" % (name, features[name].shape))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    global_step = tf.train.get_or_create_global_step()
    ##### Classification objective
    label_ids = features["label_ids"]
    label_ids = tf.reshape(label_ids, [-1])

    if unsup_ratio > 0 and "ori_input_ids" in features:
      input_ids = tf.concat([
          features["input_ids"],
          features["ori_input_ids"],
          features["aug_input_ids"]], 0)
      input_mask = tf.concat([
          features["input_mask"],
          features["ori_input_mask"],
          features["aug_input_mask"]], 0)
      input_type_ids = tf.concat([
          features["input_type_ids"],
          features["ori_input_type_ids"],
          features["aug_input_type_ids"]], 0)
    else:
      input_ids = features["input_ids"]
      input_mask = features["input_mask"]
      input_type_ids = features["input_type_ids"]

    (sup_loss, unsup_loss, logits,
     per_example_loss, loss_mask,
     tsa_threshold,
     unsup_loss_mask, correct_label_probs) = create_model(
         bert_config=bert_config,
         is_training=is_training,
         input_ids=input_ids,
         input_mask=input_mask,
         input_type_ids=input_type_ids,
         labels=label_ids,
         num_labels=num_labels,
         use_one_hot_embeddings=use_one_hot_embeddings,
         tsa=tsa,
         unsup_ratio=unsup_ratio,
         global_step=global_step,
         num_train_steps=num_train_steps,
         )

    ##### Aggregate losses into total_loss
    metric_dict = {}

    # number of correct predictions
    predictions = tf.argmax(logits, axis=-1, output_type=label_ids.dtype)
    is_correct = tf.to_float(tf.equal(predictions, label_ids))
    acc = tf.reduce_mean(is_correct)
    # add sup. metrics to dict
    metric_dict["sup/loss"] = sup_loss
    metric_dict["sup/accu"] = acc
    metric_dict["sup/correct_cat_probs"] = correct_label_probs
    metric_dict["sup/tsa_threshold"] = tsa_threshold

    metric_dict["sup/sup_trained_ratio"] = tf.reduce_mean(loss_mask)
    total_loss = sup_loss

    if unsup_ratio > 0 and uda_coeff > 0 and "input_ids" in features:
      total_loss += uda_coeff * unsup_loss
      metric_dict["unsup/loss"] = unsup_loss

    if unsup_loss_mask is not None:
      metric_dict["unsup/high_prob_ratio"] = tf.reduce_mean(unsup_loss_mask)

    ##### Initialize variables with pre-trained models
    tvars = tf.trainable_variables()

    scaffold_fn = None
    if init_checkpoint:
      (assignment_map,
       initialized_variable_names) = get_assignment_map_from_checkpoint(
           tvars, init_checkpoint)
      if use_tpu:
        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    else:
      initialized_variable_names = {}

    if print_structure:
      tf.logging.info("**** Trainable Variables ****")
      for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
          init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)

    ##### Construct TPU Estimator Spec based on the specific mode
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      ## Create optimizer for training
      train_op, curr_lr = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps,
          use_tpu, clip_norm, global_step)
      metric_dict["learning_rate"] = curr_lr

      ## Create host_call for training
      host_call = tpu_utils.construct_scalar_host_call(
          metric_dict=metric_dict,
          model_dir=params["model_dir"],
          prefix="training/",
          reduce_fn=tf.reduce_mean)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          host_call=host_call,
          scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.EVAL:

      def clas_metric_fn(per_example_loss, label_ids, logits):
        ## classification loss & accuracy
        loss = tf.metrics.mean(per_example_loss)

        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(label_ids, predictions)

        ret_dict = {
            "eval_classify_loss": loss,
            "eval_classify_accuracy": accuracy
        }

        return ret_dict

      eval_metrics = (clas_metric_fn, [per_example_loss, label_ids, logits])

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn

