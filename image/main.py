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
"""UDA on CIFAR-10 and SVHN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import time
import json
import functools

import numpy as np

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf

from randaugment import custom_ops as ops
import data
import utils

from randaugment.wrn import build_wrn_model
from randaugment.shake_drop import build_shake_drop_model
from randaugment.shake_shake import build_shake_shake_model


# TPU related
flags.DEFINE_string(
    "master", default=None,
    help="the TPU address. This should be set when using Cloud TPU")
flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
flags.DEFINE_string(
    "gcp_project", default=None,
    help="Project name for the Cloud TPU-enabled project. If not specified, "
    "we will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string(
    "tpu_zone", default=None,
    help="GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_bool(
    "use_tpu", default=False,
    help="Use TPUs rather than GPU/CPU.")
flags.DEFINE_enum(
    "task_name", "cifar10",
    enum_values=["cifar10", "svhn"],
    help="The task to use")

# UDA config:
flags.DEFINE_integer(
    "sup_size", default=4000,
    help="Number of supervised pairs to use. "
    "-1: all training samples. 4000: 4000 supervised examples.")
flags.DEFINE_integer(
    "aug_copy", default=0,
    help="Number of different augmented data generated.")
flags.DEFINE_integer(
    "unsup_ratio", default=0,
    help="The ratio between batch size of unlabeled data and labeled data, "
    "i.e., unsup_ratio * train_batch_size is the batch_size for unlabeled data."
    "Do not use the unsupervised objective if set to 0.")
flags.DEFINE_enum(
    "tsa", "",
    enum_values=["", "linear_schedule", "log_schedule", "exp_schedule"],
    help="anneal schedule of training signal annealing. "
    "tsa='' means not using TSA. See the paper for other schedules.")
flags.DEFINE_float(
    "uda_confidence_thresh", default=-1,
    help="The threshold on predicted probability on unsupervised data. If set,"
    "UDA loss will only be calculated on unlabeled examples whose largest"
    "probability is larger than the threshold")
flags.DEFINE_float(
    "uda_softmax_temp", -1,
    help="The temperature of the Softmax when making prediction on unlabeled"
    "examples. -1 means to use normal Softmax")
flags.DEFINE_float(
    "ent_min_coeff", default=0,
    help="")
flags.DEFINE_integer(
    "unsup_coeff", default=1,
    help="The coefficient on the UDA loss. "
    "setting unsup_coeff to 1 works for most settings. "
    "When you have extermely few samples, consider increasing unsup_coeff")
flags.DEFINE_float(
    'moving_average_decay', default=0.9999,
    help=('Moving average decay rate.'))

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string(
    "data_dir", default=None,
    help="Path to data directory containing `*.tfrecords`.")
flags.DEFINE_string(
    "model_dir", default=None,
    help="model dir of the saved checkpoints.")
flags.DEFINE_bool(
    "do_train", default=True,
    help="Whether to run training.")
flags.DEFINE_bool(
    "do_eval", default=False,
    help="Whether to run eval on the test set.")
flags.DEFINE_integer(
    "dev_size", default=-1,
    help="dev set size.")
flags.DEFINE_bool(
    "verbose", default=False,
    help="Whether to print additional information.")

# Training config
flags.DEFINE_integer(
    "train_batch_size", default=32,
    help="Size of train batch.")
flags.DEFINE_integer(
    "eval_batch_size", default=8,
    help="Size of evalation batch.")
flags.DEFINE_integer(
    "train_steps", default=100000,
    help="Total number of training steps.")
flags.DEFINE_integer(
    "iterations", default=10000,
    help="Number of iterations per repeat loop.")
flags.DEFINE_integer(
    "save_steps", default=10000,
    help="number of steps for model checkpointing.")
flags.DEFINE_integer(
    "max_save", default=10,
    help="Maximum number of checkpoints to save.")

# Model config
flags.DEFINE_enum(
    "model_name", default="wrn",
    enum_values=["wrn", "shake_shake_32", "shake_shake_96", "shake_shake_112", "pyramid_net"],
    help="Name of the model")
flags.DEFINE_integer(
    "num_classes", default=10,
    help="Number of categories for classification.")
flags.DEFINE_integer(
    "wrn_size", default=32,
    help="The size of WideResNet. It should be set to 32 for WRN-28-2"
    "and should be set to 160 for WRN-28-10")

# Optimization config
flags.DEFINE_float(
    "learning_rate", default=0.03,
    help="Maximum learning rate.")
flags.DEFINE_float(
    "weight_decay_rate", default=5e-4,
    help="Weight decay rate.")
flags.DEFINE_integer(
    "warmup_steps", default=0,
    help="Number of steps for linear lr warmup.")



FLAGS = tf.flags.FLAGS

arg_scope = tf.contrib.framework.arg_scope


def get_tsa_threshold(schedule, global_step, num_train_steps, start, end):
  step_ratio = tf.to_float(global_step) / tf.to_float(num_train_steps)
  if schedule == "linear_schedule":
    coeff = step_ratio
  elif schedule == "exp_schedule":
    scale = 5
    # [exp(-5), exp(0)] = [1e-2, 1]
    coeff = tf.exp((step_ratio - 1) * scale)
  elif schedule == "log_schedule":
    scale = 5
    # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
    coeff = 1 - tf.exp((-step_ratio) * scale)
  return coeff * (end - start) + start


def setup_arg_scopes(is_training):
  """Sets up the argscopes that will be used when building an image model.

  Args:
    is_training: Is the model training or not.

  Returns:
    Arg scopes to be put around the model being constructed.
  """

  batch_norm_decay = 0.9
  batch_norm_epsilon = 1e-5
  batch_norm_params = {
      # Decay for the moving averages.
      "decay": batch_norm_decay,
      # epsilon to prevent 0s in variance.
      "epsilon": batch_norm_epsilon,
      "scale": True,
      # collection containing the moving mean and moving variance.
      "is_training": is_training,
  }

  scopes = []

  scopes.append(arg_scope([ops.batch_norm], **batch_norm_params))
  return scopes


def build_model(inputs, num_classes, is_training, update_bn, hparams):
  """Constructs the vision model being trained/evaled.

  Args:
    inputs: input features/images being fed to the image model build built.
    num_classes: number of output classes being predicted.
    is_training: is the model training or not.
    hparams: additional hyperparameters associated with the image model.

  Returns:
    The logits of the image model.
  """
  scopes = setup_arg_scopes(is_training)

  try:
      from contextlib import nested
  except ImportError:
      from contextlib import ExitStack, contextmanager

      @contextmanager
      def nested(*contexts):
          with ExitStack() as stack:
              for ctx in contexts:
                  stack.enter_context(ctx)
              yield contexts

  with nested(*scopes):
    if hparams.model_name == "pyramid_net":
      logits = build_shake_drop_model(
          inputs, num_classes, is_training)
    elif hparams.model_name == "wrn":
      logits = build_wrn_model(
          inputs, num_classes, hparams.wrn_size, update_bn)
    elif hparams.model_name == "shake_shake":
      logits = build_shake_shake_model(
          inputs, num_classes, hparams, is_training)

  return logits


def _kl_divergence_with_logits(p_logits, q_logits):
  p = tf.nn.softmax(p_logits)
  log_p = tf.nn.log_softmax(p_logits)
  log_q = tf.nn.log_softmax(q_logits)

  kl = tf.reduce_sum(p * (log_p - log_q), -1)
  return kl


def anneal_sup_loss(sup_logits, sup_labels, sup_loss, global_step, metric_dict):
  tsa_start = 1. / FLAGS.num_classes
  eff_train_prob_threshold = get_tsa_threshold(
      FLAGS.tsa, global_step, FLAGS.train_steps,
      tsa_start, end=1)

  one_hot_labels = tf.one_hot(
      sup_labels, depth=FLAGS.num_classes, dtype=tf.float32)
  sup_probs = tf.nn.softmax(sup_logits, axis=-1)
  correct_label_probs = tf.reduce_sum(
      one_hot_labels * sup_probs, axis=-1)
  larger_than_threshold = tf.greater(
      correct_label_probs, eff_train_prob_threshold)
  loss_mask = 1 - tf.cast(larger_than_threshold, tf.float32)
  loss_mask = tf.stop_gradient(loss_mask)
  sup_loss = sup_loss * loss_mask
  avg_sup_loss = (tf.reduce_sum(sup_loss) /
                  tf.maximum(tf.reduce_sum(loss_mask), 1))
  metric_dict["sup/sup_trained_ratio"] = tf.reduce_mean(loss_mask)
  metric_dict["sup/eff_train_prob_threshold"] = eff_train_prob_threshold
  return sup_loss, avg_sup_loss


def _scaffold_fn(restore_vars_dict):
  saver = tf.train.Saver(restore_vars_dict, max_to_keep=10000)
  return tf.train.Scaffold(saver=saver)


def get_ent(logits, return_mean=True):
  log_prob = tf.nn.log_softmax(logits, axis=-1)
  prob = tf.exp(log_prob)
  ent = tf.reduce_sum(-prob * log_prob, axis=-1)
  if return_mean:
    ent = tf.reduce_mean(ent)
  return ent


def get_model_fn(hparams):
  def model_fn(features, labels, mode, params):
    sup_labels = tf.reshape(features["label"], [-1])

    #### Configuring the optimizer
    global_step = tf.train.get_global_step()
    metric_dict = {}
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    if FLAGS.unsup_ratio > 0 and is_training:
      all_images = tf.concat([features["image"],
                              features["ori_image"],
                              features["aug_image"]], 0)
    else:
      all_images = features["image"]

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
      all_logits = build_model(
          inputs=all_images,
          num_classes=FLAGS.num_classes,
          is_training=is_training,
          update_bn=True and is_training,
          hparams=hparams,
      )

      sup_bsz = tf.shape(features["image"])[0]
      sup_logits = all_logits[:sup_bsz]

      sup_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=sup_labels,
          logits=sup_logits)
      sup_prob = tf.nn.softmax(sup_logits, axis=-1)
      metric_dict["sup/pred_prob"] = tf.reduce_mean(
          tf.reduce_max(sup_prob, axis=-1))
    if FLAGS.tsa:
      sup_loss, avg_sup_loss = anneal_sup_loss(sup_logits, sup_labels, sup_loss,
                                               global_step, metric_dict)
    else:
      avg_sup_loss = tf.reduce_mean(sup_loss)
    total_loss = avg_sup_loss

    if FLAGS.unsup_ratio > 0 and is_training:
      aug_bsz = tf.shape(features["ori_image"])[0]

      ori_logits = all_logits[sup_bsz : sup_bsz + aug_bsz]
      aug_logits = all_logits[sup_bsz + aug_bsz:]
      if FLAGS.uda_softmax_temp != -1:
        ori_logits_tgt = ori_logits / FLAGS.uda_softmax_temp
      else:
        ori_logits_tgt = ori_logits
      ori_prob = tf.nn.softmax(ori_logits, axis=-1)
      aug_prob = tf.nn.softmax(aug_logits, axis=-1)
      metric_dict["unsup/ori_prob"] = tf.reduce_mean(
          tf.reduce_max(ori_prob, axis=-1))
      metric_dict["unsup/aug_prob"] = tf.reduce_mean(
          tf.reduce_max(aug_prob, axis=-1))

      aug_loss = _kl_divergence_with_logits(
          p_logits=tf.stop_gradient(ori_logits_tgt),
          q_logits=aug_logits)

      if FLAGS.uda_confidence_thresh != -1:
        ori_prob = tf.nn.softmax(ori_logits, axis=-1)
        largest_prob = tf.reduce_max(ori_prob, axis=-1)
        loss_mask = tf.cast(tf.greater(
            largest_prob, FLAGS.uda_confidence_thresh), tf.float32)
        metric_dict["unsup/high_prob_ratio"] = tf.reduce_mean(loss_mask)
        loss_mask = tf.stop_gradient(loss_mask)
        aug_loss = aug_loss * loss_mask
        metric_dict["unsup/high_prob_loss"] = tf.reduce_mean(aug_loss)

      if FLAGS.ent_min_coeff > 0:
        ent_min_coeff = FLAGS.ent_min_coeff
        metric_dict["unsup/ent_min_coeff"] = ent_min_coeff
        per_example_ent = get_ent(ori_logits)
        ent_min_loss = tf.reduce_mean(per_example_ent)
        total_loss = total_loss + ent_min_coeff * ent_min_loss

      avg_unsup_loss = tf.reduce_mean(aug_loss)
      total_loss += FLAGS.unsup_coeff * avg_unsup_loss
      metric_dict["unsup/loss"] = avg_unsup_loss

    total_loss = utils.decay_weights(
        total_loss,
        FLAGS.weight_decay_rate)

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info("#params: {}".format(num_params))

    if FLAGS.verbose:
      format_str = "{{:<{0}s}}\t{{}}".format(
          max([len(v.name) for v in tf.trainable_variables()]))
      for v in tf.trainable_variables():
        tf.logging.info(format_str.format(v.name, v.get_shape()))
    if FLAGS.moving_average_decay > 0.:
      ema = tf.train.ExponentialMovingAverage(
          decay=FLAGS.moving_average_decay)
      ema_vars = utils.get_all_variable()

    #### Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
      if FLAGS.moving_average_decay > 0:
        restore_vars_dict = ema.variables_to_restore(ema_vars)
        scaffold_fn = functools.partial(
            _scaffold_fn,
            restore_vars_dict=restore_vars_dict) if FLAGS.moving_average_decay > 0 else None
      else:
        scaffold_fn = None

      #### Metric function for classification
      def metric_fn(per_example_loss, label_ids, logits):
        # classification loss & accuracy
        loss = tf.metrics.mean(per_example_loss)

        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(label_ids, predictions)

        ret_dict = {
            "eval/classify_loss": loss,
            "eval/classify_accuracy": accuracy
        }

        return ret_dict

      eval_metrics = (metric_fn, [sup_loss, sup_labels, sup_logits])

      #### Constucting evaluation TPUEstimatorSpec.
      eval_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn,
      )

      return eval_spec

    # increase the learning rate linearly
    if FLAGS.warmup_steps > 0:
      warmup_lr = tf.to_float(global_step) / tf.to_float(FLAGS.warmup_steps) \
                  * FLAGS.learning_rate
    else:
      warmup_lr = 0.0

    # decay the learning rate using the cosine schedule
    lrate = tf.clip_by_value(tf.to_float(global_step-FLAGS.warmup_steps) / (FLAGS.train_steps-FLAGS.warmup_steps), 0, 1)
    decay_lr = FLAGS.learning_rate * tf.cos(lrate * (7. / 8) * np.pi / 2)

    learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                             warmup_lr, decay_lr)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=0.9,
        use_nesterov=True)

    if FLAGS.use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    grads_and_vars = optimizer.compute_gradients(total_loss)
    gradients, variables = zip(*grads_and_vars)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.apply_gradients(
          zip(gradients, variables), global_step=tf.train.get_global_step())
    if FLAGS.moving_average_decay > 0:
      with tf.control_dependencies([train_op]):
        train_op = ema.apply(ema_vars)

    #### Creating training logging hook
    # compute accuracy
    sup_pred = tf.argmax(sup_logits, axis=-1, output_type=sup_labels.dtype)
    is_correct = tf.to_float(tf.equal(sup_pred, sup_labels))
    acc = tf.reduce_mean(is_correct)
    metric_dict["sup/sup_loss"] = avg_sup_loss
    metric_dict["training/loss"] = total_loss
    metric_dict["sup/acc"] = acc
    metric_dict["training/lr"] = learning_rate
    metric_dict["training/step"] = global_step

    if not FLAGS.use_tpu:
      log_info = ("step [{training/step}] lr {training/lr:.6f} "
                  "loss {training/loss:.4f} "
                  "sup/acc {sup/acc:.4f} sup/loss {sup/sup_loss:.6f} ")
      if FLAGS.unsup_ratio > 0:
        log_info += "unsup/loss {unsup/loss:.6f} "
      formatter = lambda kwargs: log_info.format(**kwargs)
      logging_hook = tf.train.LoggingTensorHook(
          tensors=metric_dict,
          every_n_iter=FLAGS.iterations,
          formatter=formatter)
      training_hooks = [logging_hook]
      #### Constucting training TPUEstimatorSpec.
      train_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, loss=total_loss, train_op=train_op,
          training_hooks=training_hooks)
    else:
      #### Constucting training TPUEstimatorSpec.
      host_call = utils.construct_scalar_host_call(
          metric_dict=metric_dict,
          model_dir=params["model_dir"],
          prefix="",
          reduce_fn=tf.reduce_mean)
      train_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, loss=total_loss, train_op=train_op,
          host_call=host_call)

    return train_spec

  return model_fn


def train(hparams):
  ##### Create input function
  if FLAGS.unsup_ratio == 0:
    FLAGS.aug_copy = 0
  else:
    assert FLAGS.aug_copy > 0, "Please specify aug_copy"
  if FLAGS.dev_size != -1:
    FLAGS.do_train = True
    FLAGS.do_eval = True
  if FLAGS.do_train:
    train_input_fn = data.get_input_fn(
        data_dir=FLAGS.data_dir,
        split="train",
        task_name=FLAGS.task_name,
        sup_size=FLAGS.sup_size,
        unsup_ratio=FLAGS.unsup_ratio,
        aug_copy=FLAGS.aug_copy,
    )

  if FLAGS.do_eval:
    if FLAGS.dev_size != -1:
      eval_input_fn = data.get_input_fn(
          data_dir=FLAGS.data_dir,
          split="dev",
          task_name=FLAGS.task_name,
          sup_size=FLAGS.dev_size,
          unsup_ratio=0,
          aug_copy=0)
      eval_size = FLAGS.dev_size
    else:
      eval_input_fn = data.get_input_fn(
          data_dir=FLAGS.data_dir,
          split="test",
          task_name=FLAGS.task_name,
          sup_size=-1,
          unsup_ratio=0,
          aug_copy=0)
      if FLAGS.task_name == "cifar10":
        eval_size = 10000
      elif FLAGS.task_name == "svhn":
        eval_size = 26032
      else:
        assert False, "You need to specify the size of your test set."
    eval_steps = eval_size // FLAGS.eval_batch_size

  ##### Get model function
  model_fn = get_model_fn(hparams)
  estimator = utils.get_TPU_estimator(FLAGS, model_fn)

  #### Training
  if FLAGS.dev_size != -1:
    tf.logging.info("***** Running training and validation *****")
    tf.logging.info("  Supervised batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Unsupervised batch size = %d",
                    FLAGS.train_batch_size * FLAGS.unsup_ratio)
    tf.logging.info("  Num train steps = %d", FLAGS.train_steps)
    curr_step = 0
    while True:
      if curr_step >= FLAGS.train_steps:
        break
      tf.logging.info("Current step {}".format(curr_step))
      train_step = min(FLAGS.save_steps, FLAGS.train_steps - curr_step)
      estimator.train(input_fn=train_input_fn, steps=train_step)
      estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
      curr_step += FLAGS.save_steps
  else:
    if FLAGS.do_train:
      tf.logging.info("***** Running training *****")
      tf.logging.info("  Supervised batch size = %d", FLAGS.train_batch_size)
      tf.logging.info("  Unsupervised batch size = %d",
                      FLAGS.train_batch_size * FLAGS.unsup_ratio)
      estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)
    if FLAGS.do_eval:
      tf.logging.info("***** Running evaluation *****")
      results = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
      tf.logging.info(">> Results:")
      for key in results.keys():
        tf.logging.info("  %s = %s", key, str(results[key]))
        results[key] = results[key].item()
      acc = results["eval/classify_accuracy"]
      with tf.gfile.Open("{}/results.txt".format(FLAGS.model_dir), "w") as ouf:
        ouf.write(str(acc))


def main(_):

  if FLAGS.do_train:
    tf.gfile.MakeDirs(FLAGS.model_dir)
    flags_dict = tf.app.flags.FLAGS.flag_values_dict()
    with tf.gfile.Open(os.path.join(FLAGS.model_dir, "FLAGS.json"), "w") as ouf:
      json.dump(flags_dict, ouf)
  hparams = tf.contrib.training.HParams()

  if FLAGS.model_name == "wrn":
    hparams.add_hparam("model_name", "wrn")
    hparams.add_hparam("wrn_size", FLAGS.wrn_size)
  elif FLAGS.model_name == "shake_shake_32":
    hparams.add_hparam("model_name", "shake_shake")
    hparams.add_hparam("shake_shake_widen_factor", 2)
  elif FLAGS.model_name == "shake_shake_96":
    hparams.add_hparam("model_name", "shake_shake")
    hparams.add_hparam("shake_shake_widen_factor", 6)
  elif FLAGS.model_name == "shake_shake_112":
    hparams.add_hparam("model_name", "shake_shake")
    hparams.add_hparam("shake_shake_widen_factor", 7)
  elif FLAGS.model_name == "pyramid_net":
    hparams.add_hparam("model_name", "pyramid_net")
  else:
    raise ValueError("Not Valid Model Name: %s" % FLAGS.model_name)

  train(hparams)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
