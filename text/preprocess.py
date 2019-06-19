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
"""Preprocessing for text classifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
from absl import app
from absl import flags

import numpy as np
import tensorflow as tf

# from augmentation import aug_policy
from augmentation import sent_level_augment
from augmentation import word_level_augment
from utils import raw_data_utils
from utils import tokenization


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "task_name", "IMDB", "The name of the task to train.")

flags.DEFINE_string(
    "raw_data_dir", None, "Data directory of the raw data")

flags.DEFINE_string(
    "output_base_dir", None, "Data directory of the processed data")

flags.DEFINE_string(
    "aug_ops", "bt-0.9", "augmentation method")

flags.DEFINE_integer(
    "aug_copy_num", -1,
    help="We generate multiple augmented examples for one"
    "unlabeled example, aug_copy_num is the index of the generated augmented"
    "example")

flags.DEFINE_integer(
    "max_seq_length", 512,
    help="The maximum total sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "sup_size", -1, "size of the labeled set")

flags.DEFINE_bool(
    "trunc_keep_right", True,
    help="Whether to keep the right part when truncate a sentence.")

flags.DEFINE_enum(
    "data_type", default="sup",
    enum_values=["sup", "unsup"],
    help="Which preprocess task to perform.")

flags.DEFINE_string(
    "sub_set", "train",
    "Which sub_set to preprocess. The sub_set can be train, dev and unsup_in")

flags.DEFINE_string(
    "vocab_file", "", "The path of the vocab file of BERT.")

flags.DEFINE_bool(
    "do_lower_case", True, "Whether to use uncased text for BERT.")

flags.DEFINE_string(
    "back_translation_dir", "", "Directory for back translated sentence.")

flags.DEFINE_integer(
    "replicas", 1,
    "An argument for parallel preprocessing. For example, when replicas=3,"
    "we divide the data into three parts, and only process one part"
    "according to the worker_id.")

flags.DEFINE_integer(
    "worker_id", 0,
    "An argument for parallel preprocessing. See 'replicas' for more details")


def get_data_for_worker(examples, replicas, worker_id):
  data_per_worker = len(examples) // replicas
  remainder = len(examples) - replicas * data_per_worker
  if worker_id < remainder:
    start = (data_per_worker + 1) * worker_id
    end = (data_per_worker + 1) * (worker_id + 1)
  else:
    start = data_per_worker * worker_id + remainder
    end = data_per_worker * (worker_id + 1) + remainder
  if worker_id == replicas - 1:
    assert end == len(examples)
  tf.logging.info("processing data from {:d} to {:d}".format(start, end))
  examples = examples[start: end]
  return examples, start, end


def build_vocab(examples):
  vocab = {}
  def add_to_vocab(word_list):
    for word in word_list:
      if word not in vocab:
        vocab[word] = len(vocab)
  for i in range(len(examples)):
    add_to_vocab(examples[i].word_list_a)
    if examples[i].text_b:
      add_to_vocab(examples[i].word_list_b)
  return vocab


def get_data_stats(data_stats_dir, sub_set, sup_size, replicas, examples):
  data_stats_dir = "{}/{}".format(data_stats_dir, sub_set)
  keys = ["tf_idf", "idf"]
  all_exist = True
  for key in keys:
    data_stats_path = "{}/{}.json".format(data_stats_dir, key)
    if not tf.gfile.Exists(data_stats_path):
      all_exist = False
      tf.logging.info("Not exist: {}".format(data_stats_path))
  if all_exist:
    tf.logging.info("loading data stats from {:s}".format(data_stats_dir))
    data_stats = {}
    for key in keys:
      with tf.gfile.Open(
          "{}/{}.json".format(data_stats_dir, key)) as inf:
        data_stats[key] = json.load(inf)
  else:
    assert sup_size == -1, "should use the complete set to get tf_idf"
    assert replicas == 1, "should use the complete set to get tf_idf"
    data_stats = word_level_augment.get_data_stats(examples)
    tf.gfile.MakeDirs(data_stats_dir)
    for key in keys:
      with tf.gfile.Open("{}/{}.json".format(data_stats_dir, key), "w") as ouf:
        json.dump(data_stats[key], ouf)
    tf.logging.info("dumped data stats to {:s}".format(data_stats_dir))
  return data_stats


def tokenize_examples(examples, tokenizer):
  tf.logging.info("tokenizing examples")
  for i in range(len(examples)):
    examples[i].word_list_a = tokenizer.tokenize_to_word(examples[i].text_a)
    if examples[i].text_b:
      examples[i].word_list_b = tokenizer.tokenize_to_word(examples[i].text_b)
    if i % 10000 == 0:
      tf.logging.info("finished tokenizing example {:d}".format(i))
  return examples


def convert_examples_to_features(
    examples, label_list, seq_length, tokenizer, trunc_keep_right,
    data_stats=None, aug_ops=None):
  """convert examples to features."""

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tf.logging.info("number of examples to process: {}".format(len(examples)))

  features = []

  if aug_ops:
    tf.logging.info("building vocab")
    word_vocab = build_vocab(examples)
    examples = word_level_augment.word_level_augment(
        examples, aug_ops, word_vocab, data_stats
    )

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("processing {:d}".format(ex_index))
    tokens_a = tokenizer.tokenize_to_wordpiece(example.word_list_a)
    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize_to_wordpiece(example.word_list_b)

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      if trunc_keep_right:
        _truncate_seq_pair_keep_right(tokens_a, tokens_b, seq_length - 3)
      else:
        _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > seq_length - 2:
        if trunc_keep_right:
          tokens_a = tokens_a[-(seq_length - 2):]
        else:
          tokens_a = tokens_a[0:(seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        input_type_ids.append(1)
      tokens.append("[SEP]")
      input_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    label_id = label_map[example.label]
    if ex_index < 1:
      tf.logging.info("*** Example ***")
      tf.logging.info("guid: %s" % (example.guid))
      # st = " ".join([str(x) for x in tokens])
      st = ""
      for x in tokens:
        if isinstance(x, unicode):
          st += x.encode("ascii", "replace") + " "
        else:
          st += str(x) + " "
      tf.logging.info("tokens: %s" % st)
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info(
          "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
      tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    features.append(
        InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids,
            label_id=label_id))
  return features


def _create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, input_type_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids
    self.label_id = label_id

  def get_dict_features(self):
    return {
        "input_ids": _create_int_feature(self.input_ids),
        "input_mask": _create_int_feature(self.input_mask),
        "input_type_ids": _create_int_feature(self.input_type_ids),
        "label_ids": _create_int_feature([self.label_id])
    }


class PairedUnsupInputFeatures(object):
  """Features for paired unsup data."""

  def __init__(self, ori_input_ids, ori_input_mask, ori_input_type_ids,
               aug_input_ids, aug_input_mask, aug_input_type_ids):
    self.ori_input_ids = ori_input_ids
    self.ori_input_mask = ori_input_mask
    self.ori_input_type_ids = ori_input_type_ids
    self.aug_input_ids = aug_input_ids
    self.aug_input_mask = aug_input_mask
    self.aug_input_type_ids = aug_input_type_ids

  def get_dict_features(self):
    return {
        "ori_input_ids": _create_int_feature(self.ori_input_ids),
        "ori_input_mask": _create_int_feature(self.ori_input_mask),
        "ori_input_type_ids": _create_int_feature(self.ori_input_type_ids),
        "aug_input_ids": _create_int_feature(self.aug_input_ids),
        "aug_input_mask": _create_int_feature(self.aug_input_mask),
        "aug_input_type_ids": _create_int_feature(self.aug_input_type_ids),
    }


def obtain_tfrecord_writer(data_path, worker_id, shard_cnt):
  tfrecord_writer = tf.python_io.TFRecordWriter(
      os.path.join(
          data_path,
          "tf_examples.tfrecord.{:d}.{:d}".format(worker_id, shard_cnt)))
  return tfrecord_writer


def dump_tfrecord(features, data_path, worker_id=None, max_shard_size=4096):
  """Dump tf record."""
  if not tf.gfile.Exists(data_path):
    tf.gfile.MakeDirs(data_path)
  tf.logging.info("dumping TFRecords")
  np.random.shuffle(features)
  shard_cnt = 0
  shard_size = 0
  tfrecord_writer = obtain_tfrecord_writer(data_path, worker_id, shard_cnt)
  for feature in features:
    tf_example = tf.train.Example(
        features=tf.train.Features(feature=feature.get_dict_features()))
    if shard_size >= max_shard_size:
      tfrecord_writer.close()
      shard_cnt += 1
      tfrecord_writer = obtain_tfrecord_writer(data_path, worker_id, shard_cnt)
      shard_size = 0
    shard_size += 1
    tfrecord_writer.write(tf_example.SerializeToString())
  tfrecord_writer.close()


def get_data_by_size_lim(train_examples, processor, sup_size):
  """Deterministicly get a dataset with only sup_size examples."""
  # Assuming sup_size < number of labeled data and
  # that there are same number of examples for each category
  assert sup_size % len(processor.get_labels()) == 0
  per_label_size = sup_size // len(processor.get_labels())
  per_label_examples = {}
  for i in range(len(train_examples)):
    label = train_examples[i].label
    if label not in per_label_examples:
      per_label_examples[label] = []
    per_label_examples[label] += [train_examples[i]]

  for label in processor.get_labels():
    assert len(per_label_examples[label]) >= per_label_size, (
        "label {} only has {} examples while the limit"
        "is {}".format(label, len(per_label_examples[label]), per_label_size))

  new_train_examples = []
  for i in range(per_label_size):
    for label in processor.get_labels():
      new_train_examples += [per_label_examples[label][i]]
  train_examples = new_train_examples
  return train_examples


def _truncate_seq_pair_keep_right(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop(0)
    else:
      tokens_b.pop(0)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def proc_and_save_sup_data(
    processor, sub_set, raw_data_dir, sup_out_dir,
    tokenizer, max_seq_length, trunc_keep_right,
    worker_id, replicas, sup_size):
  tf.logging.info("getting examples")
  if sub_set == "train":
    examples = processor.get_train_examples(raw_data_dir)
  elif sub_set == "dev":
    examples = processor.get_dev_examples(raw_data_dir)
    assert replicas == 1, "dev set can be processsed with just one worker"
    assert sup_size == -1, "should use the full dev set"

  if sup_size != -1:
    tf.logging.info("setting number of examples to {:d}".format(
        sup_size))
    examples = get_data_by_size_lim(
        examples, processor, sup_size)
  if replicas != 1:
    if len(examples) < replicas:
      replicas = len(examples)
      if worker_id >= replicas:
        return
    examples = get_data_for_worker(
        examples, replicas, worker_id)

  tf.logging.info("processing data")
  examples = tokenize_examples(examples, tokenizer)

  features = convert_examples_to_features(
      examples, processor.get_labels(), max_seq_length, tokenizer,
      trunc_keep_right, None, None)
  dump_tfrecord(features, sup_out_dir, worker_id)


def proc_and_save_unsup_data(
    processor, sub_set,
    raw_data_dir, data_stats_dir, unsup_out_dir,
    tokenizer,
    max_seq_length, trunc_keep_right,
    aug_ops, aug_copy_num,
    worker_id, replicas):
  # print random seed just to double check that we use different random seeds
  # for different runs so that we generate different augmented examples for the same original example.
  random_seed = np.random.randint(0, 100000)
  tf.logging.info("random seed: {:d}".format(random_seed))
  np.random.seed(random_seed)
  tf.logging.info("getting examples")

  if sub_set == "train":
    ori_examples = processor.get_train_examples(raw_data_dir)
  elif sub_set.startswith("unsup"):
    ori_examples = processor.get_unsup_examples(raw_data_dir, sub_set)
  else:
    assert False
  # this is the size before spliting data for each worker
  data_total_size = len(ori_examples)
  if replicas != -1:
    ori_examples, start, end = get_data_for_worker(
        ori_examples, replicas, worker_id)
  else:
    start = 0
    end = len(ori_examples)

  tf.logging.info("getting augmented examples")
  aug_examples = copy.deepcopy(ori_examples)
  aug_examples = sent_level_augment.run_augment(
      aug_examples, aug_ops, sub_set,
      aug_copy_num,
      start, end, data_total_size)

  labels = processor.get_labels() + ["unsup"]
  tf.logging.info("processing ori examples")
  ori_examples = tokenize_examples(ori_examples, tokenizer)
  ori_features = convert_examples_to_features(
      ori_examples, labels, max_seq_length, tokenizer,
      trunc_keep_right, None, None)

  if "idf" in aug_ops:
    data_stats = get_data_stats(
        data_stats_dir, sub_set,
        -1, replicas, ori_examples)
  else:
    data_stats = None

  tf.logging.info("processing aug examples")
  aug_examples = tokenize_examples(aug_examples, tokenizer)
  aug_features = convert_examples_to_features(
      aug_examples, labels, max_seq_length, tokenizer,
      trunc_keep_right, data_stats, aug_ops)

  unsup_features = []
  for ori_feat, aug_feat in zip(ori_features, aug_features):
    unsup_features.append(PairedUnsupInputFeatures(
        ori_feat.input_ids,
        ori_feat.input_mask,
        ori_feat.input_type_ids,
        aug_feat.input_ids,
        aug_feat.input_mask,
        aug_feat.input_type_ids,
        ))
  dump_tfrecord(unsup_features, unsup_out_dir, worker_id)


def main(_):


  if FLAGS.max_seq_length > 512:
    raise ValueError(
        "Cannot use sequence length {:d} because the BERT model "
        "was only trained up to sequence length {:d}".format(
            FLAGS.max_seq_length, 512))

  processor = raw_data_utils.get_processor(FLAGS.task_name)
  # Create tokenizer
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  if FLAGS.data_type == "sup":
    sup_out_dir = FLAGS.output_base_dir
    tf.logging.info("Create sup. data: subset {} => {}".format(
        FLAGS.sub_set, sup_out_dir))

    proc_and_save_sup_data(
        processor, FLAGS.sub_set, FLAGS.raw_data_dir, sup_out_dir,
        tokenizer, FLAGS.max_seq_length, FLAGS.trunc_keep_right,
        FLAGS.worker_id, FLAGS.replicas, FLAGS.sup_size,
    )
  elif FLAGS.data_type == "unsup":
    assert FLAGS.aug_ops is not None, \
        "aug_ops is required to preprocess unsupervised data."
    unsup_out_dir = os.path.join(
        FLAGS.output_base_dir,
        FLAGS.aug_ops,
        str(FLAGS.aug_copy_num))
    data_stats_dir = os.path.join(FLAGS.raw_data_dir, "data_stats")


    tf.logging.info("Create unsup. data: subset {} => {}".format(
        FLAGS.sub_set, unsup_out_dir))
    proc_and_save_unsup_data(
        processor, FLAGS.sub_set,
        FLAGS.raw_data_dir, data_stats_dir, unsup_out_dir,
        tokenizer, FLAGS.max_seq_length, FLAGS.trunc_keep_right,
        FLAGS.aug_ops, FLAGS.aug_copy_num,
        FLAGS.worker_id, FLAGS.replicas)


if __name__ == "__main__":
  app.run(main)
