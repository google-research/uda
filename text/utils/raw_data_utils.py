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
"""Load raw data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

from absl import flags

import tensorflow as tf

FLAGS = flags.FLAGS


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, raw_data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, raw_data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  def get_train_size(self):
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None, delimiter="\t"):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


def clean_web_text(st):
  """clean text."""
  st = st.replace("<br />", " ")
  st = st.replace("&quot;", "\"")
  st = st.replace("<p>", " ")
  if "<a href=" in st:
    # print("before:\n", st)
    while "<a href=" in st:
      start_pos = st.find("<a href=")
      end_pos = st.find(">", start_pos)
      if end_pos != -1:
        st = st[:start_pos] + st[end_pos + 1:]
      else:
        print("incomplete href")
        print("before", st)
        st = st[:start_pos] + st[start_pos + len("<a href=")]
        print("after", st)

    st = st.replace("</a>", "")
    # print("after\n", st)
    # print("")
  st = st.replace("\\n", " ")
  st = st.replace("\\", " ")
  # while "  " in st:
  #   st = st.replace("  ", " ")
  return st


class IMDbProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, raw_data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(raw_data_dir, "train.csv"),
                       quotechar='"'), "train")

  def get_dev_examples(self, raw_data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(raw_data_dir, "test.csv"),
                       quotechar='"'), "test")

  def get_unsup_examples(self, raw_data_dir, unsup_set):
    """See base class."""
    if unsup_set == "unsup_ext":
      return self._create_examples(
          self._read_tsv(os.path.join(raw_data_dir, "unsup_ext.csv"),
                         quotechar='"'), "unsup_ext", skip_unsup=False)
    elif unsup_set == "unsup_in":
      return self._create_examples(
          self._read_tsv(os.path.join(raw_data_dir, "train.csv"),
                         quotechar='"'), "unsup_in", skip_unsup=False)

  def get_labels(self):
    """See base class."""
    return ["pos", "neg"]

  def _create_examples(self, lines, set_type, skip_unsup=True):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      if skip_unsup and line[1] == "unsup":
        continue
      if line[1] == "unsup" and len(line[0]) < 500:
        # tf.logging.info("skipping short samples:{:s}".format(line[0]))
        continue
      guid = "%s-%s" % (set_type, line[2])
      text_a = line[0]
      label = line[1]
      text_a = clean_web_text(text_a)
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

  def get_train_size(self):
    return 25000

  def get_dev_size(self):
    return 25000


class TextClassProcessor(DataProcessor):

  def get_train_examples(self, raw_data_dir):
    """See base class."""
    examples = self._create_examples(
        self._read_tsv(os.path.join(raw_data_dir, "train.csv"),
                       quotechar="\"",
                       delimiter=","), "train")
    assert len(examples) == self.get_train_size()
    return examples

  def get_dev_examples(self, raw_data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(raw_data_dir, "test.csv"),
                       quotechar="\"",
                       delimiter=","), "test")

  def get_unsup_examples(self, raw_data_dir, unsup_set):
    """See base class."""
    if unsup_set == "unsup_in":
      return self._create_examples(
          self._read_tsv(
              os.path.join(raw_data_dir, "train.csv"),
              quotechar="\"",
              delimiter=","),
          "unsup_in", skip_unsup=False)
    else:
      return self._create_examples(
          self._read_tsv(
              os.path.join(raw_data_dir, "{:s}.csv".format(unsup_set)),
              quotechar="\"",
              delimiter=","),
          unsup_set, skip_unsup=False)

  def _create_examples(self, lines, set_type, skip_unsup=True,
                       only_unsup=False):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if skip_unsup and line[0] == "unsup":
        continue
      if only_unsup and line[0] != "unsup":
        continue
      guid = "%s-%d" % (set_type, i)
      if self.has_title:
        text_a = line[2]
        text_b = line[1]
      else:
        text_a = line[1]
        text_b = None
      label = line[0]
      text_a = clean_web_text(text_a)
      if text_b is not None:
        text_b = clean_web_text(text_b)
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class YELP2Processor(TextClassProcessor):

  def __init__(self):
    self.has_title = False

  def get_labels(self):
    """See base class."""
    return [str(i) for i in range(1, 3)]

  def get_train_size(self):
    return 560000

  def get_dev_size(self):
    return 38000


class YELP5Processor(TextClassProcessor):

  def __init__(self):
    self.has_title = False

  def get_labels(self):
    """See base class."""
    return [str(i) for i in range(1, 6)]

  def get_train_size(self):
    return 650000

  def get_dev_size(self):
    return 50000


class AMAZON2Processor(TextClassProcessor):

  def __init__(self):
    self.has_title = True

  def get_labels(self):
    """See base class."""
    return [str(i) for i in range(1, 3)]

  def get_train_size(self):
    return 3600000

  def get_dev_size(self):
    return 400000

  def get_unsup_examples(self, raw_data_dir, unsup_set):
    """See base class."""
    if unsup_set == "unsup_in":
      return self._create_examples(
          self._read_tsv(
              os.path.join(raw_data_dir, "train.csv"),
              quotechar="\"",
              delimiter=","),
          "unsup_in", skip_unsup=False)
    else:
      dir_cell = raw_data_dir[5:7]
      unsup_dir = None  # update this path if you use unsupervised data
      return self._create_examples(
          self._read_tsv(
              os.path.join(unsup_dir, "{:s}.csv".format(unsup_set)),
              quotechar="\"",
              delimiter=","),
          unsup_set, skip_unsup=False)


class AMAZON5Processor(TextClassProcessor):
  def __init__(self):
    self.has_title = True

  def get_labels(self):
    """See base class."""
    return [str(i) for i in range(1, 6)]

  def get_unsup_examples(self, raw_data_dir, unsup_set):
    """See base class."""
    if unsup_set == "unsup_in":
      return self._create_examples(
          self._read_tsv(
              os.path.join(raw_data_dir, "train.csv"),
              quotechar="\"",
              delimiter=","),
          "unsup_in", skip_unsup=False)
    else:
      dir_cell = raw_data_dir[5:7]
      unsup_dir = None  # update this path if you use unsupervised data
      return self._create_examples(
          self._read_tsv(
              os.path.join(unsup_dir, "{:s}.csv".format(unsup_set)),
              quotechar="\"",
              delimiter=","),
          unsup_set, skip_unsup=False)

  def get_train_size(self):
    return 3000000

  def get_dev_size(self):
    return 650000


class DBPediaProcessor(TextClassProcessor):

  def __init__(self):
    self.has_title = True

  def get_labels(self):
    """See base class."""
    return [str(i) for i in range(1, 15)]

  def get_train_size(self):
    return 560000

  def get_dev_size(self):
    return 70000


def get_processor(task_name):
  """get processor."""
  task_name = task_name.lower()
  processors = {
      "imdb": IMDbProcessor,
      "dbpedia": DBPediaProcessor,
      "yelp-2": YELP2Processor,
      "yelp-5": YELP5Processor,
      "amazon-2": AMAZON2Processor,
      "amazon-5": AMAZON5Processor,
  }
  processor = processors[task_name]()
  return processor

