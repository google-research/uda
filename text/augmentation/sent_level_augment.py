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
"""Sentence level augmentations: back translation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import random
from absl import flags

import numpy as np
import tensorflow as tf

from augmentation import word_level_augment
from utils import raw_data_utils


FLAGS = flags.FLAGS


def replace_with_length_check(
    ori_text, new_text,
    use_min_length,
    use_max_length_diff_ratio):
  """Use new_text if the text length satisfies several constraints."""
  if len(ori_text) < use_min_length or len(new_text) < use_min_length:
    if random.random() < 0.001:
      tf.logging.info(
          "not replacing due to short text: \n\tori: {:s}\n\tnew: {:s}\n".format(
              word_level_augment.filter_unicode(ori_text),
              word_level_augment.filter_unicode(new_text)))
    return ori_text
  length_diff_ratio = 1.0 * (len(new_text) - len(ori_text)) / len(ori_text)
  if math.fabs(length_diff_ratio) > use_max_length_diff_ratio:
    if random.random() < 0.001:
      tf.logging.info(
          ("not replacing due to too different text length:\n"
           "\tori: {:s}\n\tnew: {:s}\n".format(
               word_level_augment.filter_unicode(ori_text),
               word_level_augment.filter_unicode(new_text))))
    return ori_text
  return new_text


def back_translation(examples, aug_ops, sub_set, aug_copy_num,
                     start, end, data_total_size):
  """Run back translation."""
  use_min_length = 10
  use_max_length_diff_ratio = 0.5
  tf.logging.info("running bt augmentation")
  bt_args = aug_ops.split("-")
  temp = float(bt_args[1])

  if len(bt_args) > 2:
    assert len(bt_args) == 3
    assert float(bt_args[2]) == 1.

  if examples[0].text_b is not None:
    text_per_example = 2
  else:
    text_per_example = 1

  back_translation_file = "{:s}/{:s}/sample_{:.1f}/para/para_{:d}.txt".format(
      FLAGS.back_translation_dir, sub_set,
      temp, aug_copy_num)
  tf.logging.info("Using back translation file: {:s}".format(
      back_translation_file))

  with tf.gfile.Open(back_translation_file) as inf:
    paraphrases = inf.readlines()
  for i in range(len(paraphrases)):
    paraphrases[i] = paraphrases[i].strip()
  assert len(paraphrases) == data_total_size

  paraphrases = paraphrases[start * text_per_example : end * text_per_example]
  aug_examples = []
  aug_cnt = 0
  for i in range(len(examples)):
    ori_example = examples[i]
    text_a = replace_with_length_check(
        ori_example.text_a,
        paraphrases[i * text_per_example],
        use_min_length,
        use_max_length_diff_ratio,
        )
    if text_a == paraphrases[i * text_per_example]:
      aug_cnt += 1
    if ori_example.text_b is not None:
      text_b = replace_with_length_check(
          ori_example.text_b,
          paraphrases[i * text_per_example + 1],
          use_min_length,
          use_max_length_diff_ratio,
          )
    else:
      text_b = None

    example = raw_data_utils.InputExample(
        guid=ori_example.guid,
        text_a=text_a,
        text_b=text_b,
        label=ori_example.label)
    aug_examples += [example]
    if np.random.random() < 0.0001:
      tf.logging.info("\tori:\n\t\t{:s}\n\t\t{:s}\n\t\t{:s}\n".format(
          ori_example.text_a, ori_example.text_b, ori_example.label))
      tf.logging.info("\tnew:\n\t\t{:s}\n\t\t{:s}\n\t\t{:s}\n".format(
          example.text_a, example.text_b, example.label))
    if i % 10000 == 0:
      print("processing example # {:d}".format(i))
  tf.logging.info("applied back translation for {:.1f} percent of data".format(
      aug_cnt * 1. / len(examples) * 100))
  tf.logging.info("finishing running back translation augmentation")
  return aug_examples


def run_augment(
    examples, aug_ops, sub_set, aug_copy_num,
    start, end, dst_tot_size):
  """Sentence level augmentations. Used before augmentation."""
  if aug_ops:
    if aug_ops.startswith("bt"):
      examples = back_translation(
          examples, aug_ops, sub_set, aug_copy_num, start, end, dst_tot_size)
    else:
      pass
  return examples
