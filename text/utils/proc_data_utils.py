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
"""build datasets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import os
import string

from absl import flags

import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def get_sup_feature_specs():
  """Get supervised feature."""
  feature_specs = collections.OrderedDict()
  feature_specs["input_ids"] = tf.FixedLenFeature(
      [FLAGS.max_seq_length], tf.int64)
  feature_specs["input_mask"] = tf.FixedLenFeature(
      [FLAGS.max_seq_length], tf.int64)
  feature_specs["input_type_ids"] = tf.FixedLenFeature(
      [FLAGS.max_seq_length], tf.int64)
  feature_specs["label_ids"] = tf.FixedLenFeature(
      [1], tf.int64)
  return feature_specs


def get_unsup_feature_specs():
  """Get unsupervised feature."""
  feature_specs = collections.OrderedDict()
  feature_specs["ori_input_ids"] = tf.FixedLenFeature(
      [FLAGS.max_seq_length], tf.int64)
  feature_specs["ori_input_mask"] = tf.FixedLenFeature(
      [FLAGS.max_seq_length], tf.int64)
  feature_specs["ori_input_type_ids"] = tf.FixedLenFeature(
      [FLAGS.max_seq_length], tf.int64)
  feature_specs["aug_input_ids"] = tf.FixedLenFeature(
      [FLAGS.max_seq_length], tf.int64)
  feature_specs["aug_input_mask"] = tf.FixedLenFeature(
      [FLAGS.max_seq_length], tf.int64)
  feature_specs["aug_input_type_ids"] = tf.FixedLenFeature(
      [FLAGS.max_seq_length], tf.int64)
  return feature_specs


def get_aug_files(data_base_path, aug_ops, aug_copy):
  """get aug files."""

  sub_policy_list = aug_ops.split("+")
  total_data_files = []
  for sub_policy in sub_policy_list:
    sub_policy_data_files = []
    exist_copy_num = {}
    for copy_dir in tf.gfile.ListDirectory(os.path.join(
        data_base_path, sub_policy)):
      copy_num = int(copy_dir.strip("/"))
      if copy_num >= aug_copy:
        continue
      exist_copy_num[copy_num] = 1
      data_record_path = os.path.join(
          data_base_path, sub_policy, copy_dir, "tf_examples.tfrecord*")
      data_files = tf.contrib.slim.parallel_reader.get_data_files(
          data_record_path)
      sub_policy_data_files += data_files
    if len(exist_copy_num) < aug_copy * 0.9:
      tf.logging.info("not enough copies for aug op: {:s}".format(aug_ops))
      tf.logging.info("found files: {:s}".format(
          " ".join(sub_policy_data_files)))
      tf.logging.info("found copy: {:d} / desired copy: {:d}".format(
          len(exist_copy_num), aug_copy))
    assert len(exist_copy_num) > aug_copy * 0.9
    total_data_files += sub_policy_data_files
  np.random.shuffle(total_data_files)
  return total_data_files


def get_training_dataset(total_data_files, batch_size, num_threads, is_training,
                         shuffle_buffer_size, feature_specs):
  """build dataset from files."""
  d = tf.data.Dataset.from_tensor_slices(tf.constant(total_data_files))
  d = d.apply(
      tf.contrib.data.shuffle_and_repeat(
          buffer_size=len(total_data_files)))

  # `cycle_length` is the number of parallel files that get read.
  cycle_length = min(num_threads, len(total_data_files))

  # `sloppy` mode means that the interleaving is not exact. This adds
  # even more randomness to the training pipeline.
  d = d.apply(
      tf.contrib.data.parallel_interleave(
          tf.data.TFRecordDataset,
          sloppy=is_training,
          cycle_length=cycle_length))
  d = d.shuffle(buffer_size=shuffle_buffer_size)
  d = d.apply(
      tf.contrib.data.map_and_batch(
          lambda record: _decode_record(record, feature_specs),
          batch_size=batch_size,
          num_parallel_batches=num_threads,
          drop_remainder=is_training))
  return d


def get_evaluation_dataset(total_data_files, batch_size, feature_specs):
  """build non-repeat dataset from files."""
  d = tf.data.TFRecordDataset(total_data_files)
  d = d.apply(
      tf.contrib.data.map_and_batch(
          lambda record: _decode_record(record, feature_specs),
          batch_size=batch_size,
          num_parallel_batches=None,
          drop_remainder=True))

  return d


def evaluation_input_fn_builder(data_base_path, task, prefetch_size=1000):

  total_data_files = tf.contrib.slim.parallel_reader.get_data_files(
      os.path.join(data_base_path, "tf_examples.tfrecord*"))
  tf.logging.info("loading eval {} data from these files: {:s}".format(
      task, " ".join(total_data_files)))

  def input_fn(params):
    batch_size = params["batch_size"]

    if task == "clas":
      dataset = get_evaluation_dataset(
          total_data_files,
          batch_size,
          get_sup_feature_specs())
    else:
      assert False

    dataset = dataset.prefetch(prefetch_size)

    return dataset

  return input_fn


def training_input_fn_builder(
    sup_data_base_path=None,
    unsup_data_base_path=None,
    aug_ops=None,
    aug_copy=None,
    unsup_ratio=None,
    num_threads=8,
    shuffle_buffer_size=100000,
    prefetch_size=1000):

  sup_total_data_files = tf.contrib.slim.parallel_reader.get_data_files(
      os.path.join(sup_data_base_path, "tf_examples.tfrecord*"))

  if unsup_ratio is not None and unsup_ratio > 0:
    assert aug_ops is not None and aug_copy is not None, \
        "Require aug_ops, aug_copy to load augmented unsup data."
    assert unsup_data_base_path is not None and unsup_data_base_path != "", \
        "Require unsup_data_base_path to load unsup data. Get {}.".format(
            unsup_data_base_path)

    unsup_total_data_files = get_aug_files(
        unsup_data_base_path, aug_ops, aug_copy)

  is_training = True

  def input_fn(params):
    """The `input_fn` for TPUEstimator which generates the feature dataset."""
    sup_batch_size = params["batch_size"]
    total_batch_size = 0
    tf.logging.info("sup batch size: %d", (sup_batch_size))

    dataset_list = []

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if sup_data_base_path is not None:
      sup_dst = get_training_dataset(
          sup_total_data_files,
          sup_batch_size,
          num_threads,
          is_training,
          shuffle_buffer_size,
          get_sup_feature_specs())
      total_batch_size += sup_batch_size
      tf.logging.info("sup batch size: %d", (sup_batch_size))
      dataset_list.append(sup_dst)

      ## only consider unsupervised data when supervised data is considered
      if unsup_data_base_path is not None and FLAGS.unsup_ratio > 0:
        unsup_dst = get_training_dataset(
            unsup_total_data_files,
            sup_batch_size * unsup_ratio,
            num_threads,
            is_training,
            shuffle_buffer_size,
            get_unsup_feature_specs())
        total_batch_size += sup_batch_size * unsup_ratio * 2
        dataset_list.append(unsup_dst)
        tf.logging.info("unsup batch size: %d", (sup_batch_size * unsup_ratio))

    tf.logging.info("total sample in a batch: %d", (total_batch_size))

    def flatten_input(*features):
      """Merging multiple feature dicts resulted from zipped datasets."""
      result = {}
      for feature in features:
        for key in feature:
          assert key not in result
          result[key] = feature[key]

      return result

    if len(dataset_list) > 1:
      d = tf.data.Dataset.zip(tuple(dataset_list))
      d = d.map(flatten_input)
    else:
      d = dataset_list[0]

    # Prefetching creates a buffer to make sure there is always data to
    # read in the event of network latency variance.
    d = d.prefetch(prefetch_size)

    # TPUEstimator supports returning a dataset instead of just features.
    # It will call `make_one_shot_iterator()` and such.
    return d
  return input_fn


