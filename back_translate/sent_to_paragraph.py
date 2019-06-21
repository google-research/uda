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
"""Compose paraphrased sentences back to paragraphs.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from absl import app
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file", "", "back translated file of sentences.")
flags.DEFINE_string(
    "output_file", "", "paraphrased sentences.")
flags.DEFINE_string(
    "doc_len_file", "", "The file that records the length information.")


def main(argv):
  with tf.gfile.Open(FLAGS.input_file) as inf:
    sentences = inf.readlines()
  with tf.gfile.Open(FLAGS.doc_len_file) as inf:
    doc_len_list = json.load(inf)
  cnt = 0
  print("\n" * 2)
  print("*** printing paraphrases ***")
  with tf.gfile.Open(FLAGS.output_file, "w") as ouf:
    for i, sent_num in enumerate(doc_len_list):
      para = ""
      for _ in range(sent_num):
        para += sentences[cnt].strip() + " "
        cnt += 1
      print("paraphrase {}: {}".format(i, para))
      ouf.write(para.strip() + "\n")


if __name__ == '__main__':
  app.run(main)
