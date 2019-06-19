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
"""Runner for UDA that uses BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tensorflow as tf

import uda
from bert import modeling
from utils import proc_data_utils
from utils import raw_data_utils



flags = tf.flags
FLAGS = flags.FLAGS


##### Task
flags.DEFINE_bool(
    "do_train", True,
    help=("Whether to perform training. If both do_train and do_eval are True, "
          "we interleave training with evaluation. But on TPU Pods, "
          "it is not possible to interleave training and evaluation"))
flags.DEFINE_bool(
    "do_eval", False,
    help="Whether to perform evaluation.")

# unsupervised objective related hyperparameters
flags.DEFINE_integer(
    "unsup_ratio", 0,
    help="The ratio between batch size of unlabeled data and labeled data, "
    "The batch_size for the unsupervised loss is unsup_ratio * train_batch_size."
    "Do not use the unsupervised objective if unsup_ratio is set to 0.")
flags.DEFINE_string(
    "aug_ops", "",
    help="Augmentation operations.")
flags.DEFINE_integer(
    "aug_copy", -1,
    help="Number of different augmented data generated.")
flags.DEFINE_float(
    "uda_coeff", 1,
    help="Coefficient on the uda loss. We set it to 1 for all"
    "of our text experiments.")
flags.DEFINE_enum(
    "tsa", "",
    enum_values=["", "linear_schedule", "log_schedule", "exp_schedule"],
    help="anneal schedule of training signal annealing. "
    "tsa='' means not using TSA. See the paper for other schedules.")
flags.DEFINE_float(
    "uda_softmax_temp", -1,
    help="The temperature of the Softmax when making prediction on unlabeled"
    "examples. -1 means to use normal Softmax")
flags.DEFINE_float(
    "uda_confidence_thresh", default=-1,
    help="The threshold on predicted probability on unsupervised data. If set,"
    "UDA loss will only be calculated on unlabeled examples whose largest"
    "probability is larger than the threshold")


##### TPU/GPU related
flags.DEFINE_bool(
    "use_tpu", True,
    "Whether to use TPU or GPU/CPU.")
flags.DEFINE_string(
    "tpu_name", default=None,
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
flags.DEFINE_string(
    "master", None,
    "If using a TPU, the address of the master.")

##### Configs related to training
flags.DEFINE_string(
    "sup_train_data_dir", None,
    help="The input data dir of the supervised data. Should contain"
    "`tf_examples.tfrecord*`")
flags.DEFINE_string(
    "eval_data_dir", None,
    help="The input data dir of the evaluation data. Should contain "
    "`tf_examples.tfrecord*`")
flags.DEFINE_string(
    "unsup_data_dir", None,
    help="The input data dir of the unsupervised data. Should contain "
    "`tf_examples.tfrecord*`")
flags.DEFINE_string(
    "bert_config_file", None,
    help="The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")
flags.DEFINE_string(
    "vocab_file", None,
    help="The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string(
    "init_checkpoint", None,
    help="Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_string(
    "task_name", None,
    help="The name of the task to train.")
flags.DEFINE_string(
    "model_dir", None,
    help="The output directory where the model checkpoints will be written.")

##### Model configuration
flags.DEFINE_bool(
    "use_one_hot_embeddings", True,
    help="If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")
flags.DEFINE_integer(
    "max_seq_length", 512,
    help="The maximum total sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_float(
    "model_dropout", -1,
    help="Dropout rate for both the attention and the hidden states.")

##### Training hyper-parameters
flags.DEFINE_integer(
    "train_batch_size", 32,
    help="Batch size for the supervised objective.")
flags.DEFINE_integer(
    "eval_batch_size", 8,
    help="Base batch size for evaluation.")
flags.DEFINE_integer(
    "save_checkpoints_num", 20,
    help="How many checkpoints we save in training.")
flags.DEFINE_integer(
    "iterations_per_loop", 200,
    help="How many steps to make in each estimator call.")
flags.DEFINE_integer(
    "num_train_steps", None,
    help="Total number of training steps to perform.")


##### Optimizer hyperparameters
flags.DEFINE_float(
    "learning_rate", 2e-5,
    help="The initial learning rate for Adam.")
flags.DEFINE_integer(
    "num_warmup_steps", None,
    help="Number of warmup steps.")
flags.DEFINE_float(
    "clip_norm", 1.0,
    help="Gradient clip hyperparameter.")




def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)

  processor = raw_data_utils.get_processor(FLAGS.task_name)
  label_list = processor.get_labels()

  bert_config = modeling.BertConfig.from_json_file(
      FLAGS.bert_config_file,
      FLAGS.model_dropout)


  tf.gfile.MakeDirs(FLAGS.model_dir)

  flags_dict = tf.app.flags.FLAGS.flag_values_dict()
  with tf.gfile.Open(os.path.join(FLAGS.model_dir, "FLAGS.json"), "w") as ouf:
    json.dump(flags_dict, ouf)

  tf.logging.info("warmup steps {}/{}".format(
      FLAGS.num_warmup_steps, FLAGS.num_train_steps))

  save_checkpoints_steps = FLAGS.num_train_steps // FLAGS.save_checkpoints_num
  tf.logging.info("setting save checkpoints steps to {:d}".format(
      save_checkpoints_steps))

  FLAGS.iterations_per_loop = min(save_checkpoints_steps,
                                  FLAGS.iterations_per_loop)
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
  else:
    tpu_cluster_resolver = None
  # if not FLAGS.use_tpu and FLAGS.num_gpu > 1:
  #   train_distribute = tf.contrib.distribute.MirroredStrategy(
  #       num_gpus=FLAGS.num_gpu)
  # else:
  #   train_distribute = None

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=save_checkpoints_steps,
      keep_checkpoint_max=1000,
      # train_distribute=train_distribute,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          per_host_input_for_training=is_per_host))

  model_fn = uda.model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      clip_norm=FLAGS.clip_norm,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_one_hot_embeddings,
      num_labels=len(label_list),
      unsup_ratio=FLAGS.unsup_ratio,
      uda_coeff=FLAGS.uda_coeff,
      tsa=FLAGS.tsa,
      print_feature=False,
      print_structure=False,
  )

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      params={"model_dir": FLAGS.model_dir},
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
    tf.logging.info("  >>> sup data dir : {}".format(FLAGS.sup_train_data_dir))
    if FLAGS.unsup_ratio > 0:
      tf.logging.info("  >>> unsup data dir : {}".format(
          FLAGS.unsup_data_dir))

    train_input_fn = proc_data_utils.training_input_fn_builder(
        FLAGS.sup_train_data_dir,
        FLAGS.unsup_data_dir,
        FLAGS.aug_ops,
        FLAGS.aug_copy,
        FLAGS.unsup_ratio)

  if FLAGS.do_eval:
    tf.logging.info("  >>> dev data dir : {}".format(FLAGS.eval_data_dir))
    eval_input_fn = proc_data_utils.evaluation_input_fn_builder(
        FLAGS.eval_data_dir,
        "clas")

    eval_size = processor.get_dev_size()
    eval_steps = int(eval_size / FLAGS.eval_batch_size)

  if FLAGS.do_train and FLAGS.do_eval:
    tf.logging.info("***** Running training & evaluation *****")
    tf.logging.info("  Supervised batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Unsupervised batch size = %d",
                    FLAGS.train_batch_size * FLAGS.unsup_ratio)
    tf.logging.info("  Num steps = %d", FLAGS.num_train_steps)
    tf.logging.info("  Base evaluation batch size = %d", FLAGS.eval_batch_size)
    tf.logging.info("  Num steps = %d", eval_steps)
    best_acc = 0
    for _ in range(0, FLAGS.num_train_steps, save_checkpoints_steps):
      tf.logging.info("*** Running training ***")
      estimator.train(
          input_fn=train_input_fn,
          steps=save_checkpoints_steps)
      tf.logging.info("*** Running evaluation ***")
      dev_result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
      tf.logging.info(">> Results:")
      for key in dev_result.keys():
        tf.logging.info("  %s = %s", key, str(dev_result[key]))
        dev_result[key] = dev_result[key].item()
      best_acc = max(best_acc, dev_result["eval_classify_accuracy"])
    tf.logging.info("***** Final evaluation result *****")
    tf.logging.info("Best acc: {:.3f}\n\n".format(best_acc))
  elif FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Supervised batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Unsupervised batch size = %d",
                    FLAGS.train_batch_size * FLAGS.unsup_ratio)
    tf.logging.info("  Num steps = %d", FLAGS.num_train_steps)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)
  elif FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Base evaluation batch size = %d", FLAGS.eval_batch_size)
    tf.logging.info("  Num steps = %d", eval_steps)
    checkpoint_state = tf.train.get_checkpoint_state(FLAGS.model_dir)

    best_acc = 0
    for ckpt_path in checkpoint_state.all_model_checkpoint_paths:
      if not tf.gfile.Exists(ckpt_path + ".data-00000-of-00001"):
        tf.logging.info(
            "Warning: checkpoint {:s} does not exist".format(ckpt_path))
        continue
      tf.logging.info("Evaluating {:s}".format(ckpt_path))
      dev_result = estimator.evaluate(
          input_fn=eval_input_fn,
          steps=eval_steps,
          checkpoint_path=ckpt_path,
      )
      tf.logging.info(">> Results:")
      for key in dev_result.keys():
        tf.logging.info("  %s = %s", key, str(dev_result[key]))
        dev_result[key] = dev_result[key].item()
      best_acc = max(best_acc, dev_result["eval_classify_accuracy"])
    tf.logging.info("***** Final evaluation result *****")
    tf.logging.info("Best acc: {:.3f}\n\n".format(best_acc))


if __name__ == "__main__":
  tf.app.run()
