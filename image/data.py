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
"""Loading module of CIFAR && SVHN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf


FLAGS = flags.FLAGS


def format_sup_filename(split, sup_size=-1):
  if split == "test":
    return "test.tfrecord"
  elif split == "train" or split == "dev":
    if sup_size == -1:
      return "{}-full.tfrecord".format(split)
    else:
      return "{}-size_{:d}.tfrecord".format(split, sup_size)
  else:
    assert False


def format_unsup_filename(aug_copy_num):
  return "unsup-{:d}.tfrecord".format(aug_copy_num)


def _postprocess_example(example):
  """Convert tensor type for TPU, cast int64 into int32 and cast sparse to dense."""
  for key in list(example.keys()):
    val = example[key]
    if tf.keras.backend.is_sparse(val):
      val = tf.sparse.to_dense(val)
    if val.dtype == tf.int64:
      val = tf.to_int32(val)
    example[key] = val


def get_dataset(file_prefix_list, record_spec, task_name,
                split, per_core_bsz):

  is_training = (split == "train")
  is_training_tensor = tf.constant(is_training, dtype=tf.bool)

  def apply_normal_aug(image):
    if task_name == "cifar10":
      image = flip(image, is_training_tensor)
    image = crop(image, is_training_tensor)
    return image

  def parser(record):
    # retrieve serialized example
    example = tf.parse_single_example(
        serialized=record,
        features=record_spec)
    # reshape image back to 3D shape
    for key in example.keys():
      if "image" in key:
        example[key] = tf.reshape(example[key], [32, 32, 3])
        example[key] = apply_normal_aug(example[key])

    _postprocess_example(example)

    return example

  all_file_list = []
  for file_prefix in file_prefix_list:
    cur_file_list = tf.contrib.slim.parallel_reader.get_data_files(
        file_prefix + "*")
    all_file_list += cur_file_list
  dataset = tf.data.Dataset.from_tensor_slices(all_file_list)

  if is_training:
    dataset = dataset.shuffle(len(all_file_list)).repeat()
  # read from 4 tfrecord files in parallel
  dataset = dataset.apply(
      tf.contrib.data.parallel_interleave(
          tf.data.TFRecordDataset,
          sloppy=is_training,
          cycle_length=4))

  if is_training:
    # Shuffle and then repeat to maintain the epoch boundary
    dataset = dataset.shuffle(100000)
    dataset = dataset.repeat()

  dataset = dataset.map(parser, num_parallel_calls=32)
  dataset = dataset.batch(per_core_bsz, drop_remainder=True)

  # Safe guard the case that the shuffle buffer size for record is smaller
  # than the batch size.
  if is_training:
    dataset = dataset.shuffle(512)
  dataset = dataset.prefetch(1)

  return dataset


def flip(image, is_training):
  def func(inp):
    flips = tf.to_float(tf.random_uniform([1, 1, 1], 0, 2, tf.int32))
    flipped_inp = tf.image.flip_left_right(inp)
    return flips * flipped_inp + (1 - flips) * inp

  return tf.cond(is_training, lambda: func(image), lambda: image)


def crop(image, is_training):
  def func(inp):
    amount = 4
    pad_inp = tf.pad(inp,
                     tf.constant([[amount, amount],
                                  [amount, amount],
                                  [0, 0]]),
                     "REFLECT")
    cropped_data = tf.random_crop(pad_inp, tf.shape(image))
    return cropped_data

  return tf.cond(is_training, lambda: func(image), lambda: image)


def get_input_fn(
    data_dir, split, task_name, sup_size=-1,
    unsup_ratio=0, aug_copy=0):

  def input_fn(params):
    per_core_bsz = params["batch_size"]

    datasets = []
    # Supervised data
    filename = format_sup_filename(split, sup_size=sup_size)
    sup_record_spec = {
        "image": tf.FixedLenFeature([32 * 32 * 3], tf.float32),
        "label": tf.FixedLenFeature([1], tf.int64)
    }
    sup_file_list = [os.path.join(data_dir, filename)]
    tf.logging.info("getting supervised dataset from {} file prefixes".format(
        len(sup_file_list)))
    sup_dataset = get_dataset(
        file_prefix_list=sup_file_list,
        record_spec=sup_record_spec,
        task_name=task_name,
        split=split,
        per_core_bsz=per_core_bsz,
    )

    datasets.append(sup_dataset)

    if unsup_ratio > 0:
      aug_record_spec = {
          "ori_image": tf.FixedLenFeature([32 * 32 * 3], tf.float32),
          "aug_image": tf.FixedLenFeature([32 * 32 * 3], tf.float32),
      }
      aug_file_list = [
          os.path.join(data_dir, format_unsup_filename(i)) for i in range(aug_copy)]
      tf.logging.info(
          "getting unsupervised dataset from {} file prefixes".format(
              len(aug_file_list))
      )
      aug_dataset = get_dataset(
          file_prefix_list=aug_file_list,
          record_spec=aug_record_spec,
          task_name=task_name,
          split=split,
          per_core_bsz=per_core_bsz * unsup_ratio
      )
      datasets.append(aug_dataset)

    def flatten_input(*features):
      result = {}
      for feature in features:
        for key in feature:
          assert key not in result
          result[key] = feature[key]
      return result

    if len(datasets) > 1:
      dataset = tf.data.Dataset.zip(tuple(datasets))
      dataset = dataset.map(flatten_input)
    else:
      dataset = datasets[0]

    return dataset

  return input_fn
