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
"""Preprocess supervised data and unsupervised data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import cPickle as pickle
from six.moves import xrange  # pylint: disable=redefined-builtin

import collections
import os
import sys
import tarfile

import numpy as np

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import scipy.io
import tensorflow as tf

from autoaugment import policies as found_policies
from autoaugment import augmentation_transforms

FLAGS = flags.FLAGS

CIFAR_TARNAME = "cifar-10-python.tar.gz"
CIFAR_DOWNLOAD_URL = "https://www.cs.toronto.edu/~kriz/" + CIFAR_TARNAME
SVHN_DOWNLOAD_URL = "http://ufldl.stanford.edu/housenumbers/{}_32x32.mat"

DOWNLOAD_DATA_FOLDER = "downloaded_data"
MERGE_DATA_FOLDER = "merged_raw_data"

random_seed = np.random.randint(0, 10000)


def format_sup_filename(split, sup_size=-1):
  if split == "test":
    return "test.tfrecord"
  elif split == "train":
    if sup_size == -1:
      return "train-full.tfrecord".format(sup_size)
    else:
      return "train-size_{:d}.tfrecord".format(sup_size)


def format_unsup_filename(aug_copy_num):
  return "unsup-{:d}.tfrecord".format(aug_copy_num)


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))


def get_raw_data_filenames(split):
  """Returns the file names expected to exist in the input_dir."""
  if FLAGS.task_name == "cifar10":
    if split == "train":
      return ["data_batch_%d" % i for i in xrange(1, 6)]
    elif split == "test":
      return ["test_batch"]
  else:
    assert False
  return file_names


def read_pickle_from_file(filename):
  with tf.gfile.Open(filename, "rb") as f:
    if sys.version_info >= (3, 0):
      data_dict = pickle.load(f, encoding="bytes")
    else:
      data_dict = pickle.load(f)
  return data_dict


def obtain_tfrecord_writer(out_path, shard_cnt):
  tfrecord_writer = tf.python_io.TFRecordWriter(
      "{}.{:d}".format(out_path, shard_cnt))
  return tfrecord_writer


def save_tfrecord(example_list, out_path, max_shard_size=4096):
  shard_cnt = 0
  shard_size = 0
  record_writer = obtain_tfrecord_writer(out_path, shard_cnt)
  for example in example_list:
    if shard_size >= max_shard_size:
      record_writer.close()
      shard_cnt += 1
      record_writer = obtain_tfrecord_writer(out_path, shard_cnt)
      shard_size = 0
    shard_size += 1
    record_writer.write(example.SerializeToString())
  record_writer.close()
  tf.logging.info("saved {} examples to {}".format(len(example_list), out_path))


def save_merged_data(images, labels, split, merge_folder):
  with tf.gfile.Open(
      os.path.join(merge_folder, "{}_images.npy".format(split)), "wb") as ouf:
    np.save(ouf, images)
  with tf.gfile.Open(
      os.path.join(merge_folder, "{}_labels.npy".format(split)), "wb") as ouf:
    np.save(ouf, labels)


def download_and_extract():
  all_exist = True
  download_folder = os.path.join(FLAGS.raw_data_dir, DOWNLOAD_DATA_FOLDER)
  merge_folder = os.path.join(FLAGS.raw_data_dir, MERGE_DATA_FOLDER)
  for split in ["train", "test"]:
    for field in ["images", "labels"]:
      if not tf.gfile.Exists(os.path.join(merge_folder, "{}_{}.npy".format(
          split, field))):
        all_exist = False
  if all_exist:
    tf.logging.info("found all merged files")
    return
  tf.logging.info("downloading dataset")
  tf.gfile.MakeDirs(download_folder)
  tf.gfile.MakeDirs(merge_folder)
  if FLAGS.task_name == "cifar10":
    tf.contrib.learn.datasets.base.maybe_download(
        CIFAR_TARNAME, download_folder, CIFAR_DOWNLOAD_URL)
    tarfile.open(
        os.path.join(download_folder, CIFAR_TARNAME), "r:gz").extractall(download_folder)
    for split in ["train", "test"]:
      images_list = []
      labels_list = []
      for filename in get_raw_data_filenames(split):
        cur_data = read_pickle_from_file(
            os.path.join(download_folder, "cifar-10-batches-py", filename))
        labels_list += [cur_data[b"labels"]]
        images_list += [cur_data[b"data"]]
      images = np.concatenate(images_list, 0)
      labels = np.concatenate(labels_list, 0)
      images = images.reshape([-1, 3, 32, 32])
      images = images.transpose(0, 2, 3, 1)
      save_merged_data(images, labels, split, merge_folder)
  elif FLAGS.task_name == "svhn":
    for split in ["train", "test"]:
      tf.contrib.learn.datasets.base.maybe_download(
          "{}_32x32.mat".format(split),
          download_folder,
          SVHN_DOWNLOAD_URL.format(split))
      filename = os.path.join(download_folder, "{}_32x32.mat".format(split))
      data_dict = scipy.io.loadmat(tf.gfile.Open(filename))
      images = np.transpose(data_dict["X"], [3, 0, 1, 2])
      labels = data_dict["y"].reshape(-1)
      labels[labels == 10] = 0
      save_merged_data(images, labels, split, merge_folder)


def load_dataset():
  data = {}
  download_and_extract()
  merge_folder = os.path.join(FLAGS.raw_data_dir, MERGE_DATA_FOLDER)
  for split in ["train", "test"]:
    with tf.gfile.Open(
        os.path.join(merge_folder, "{}_images.npy".format(split))) as inf:
      images = np.load(inf)
    with tf.gfile.Open(
        os.path.join(merge_folder, "{}_labels.npy".format(split))) as inf:
      labels = np.load(inf)
    data[split] = {"images": images, "labels": labels}
  return data


def get_data_by_size_lim(images, labels, sup_size):
  if FLAGS.use_equal_split:
    chosen_images = []
    chosen_labels = []
    num_classes = 10
    assert sup_size % num_classes == 0
    cur_stats = collections.defaultdict(int)
    for i in range(len(images)):
      label = labels[i]
      if cur_stats[label] < sup_size // num_classes:
        chosen_images += [images[i]]
        chosen_labels += [labels[i]]
        cur_stats[label] += 1
    chosen_images = np.array(chosen_images)
    chosen_labels = np.array(chosen_labels)
  else:
    # use the same labeled data as in AutoAugment
    if FLAGS.task_name == "cifar10":
      chosen_images = images[:sup_size]
      chosen_labels = labels[:sup_size]
    else:
      np.random.seed(0)
      perm = np.arange(images.shape[0])
      np.random.shuffle(perm)
      chosen_images = images[perm][:sup_size]
      chosen_labels = labels[perm][:sup_size]
  return chosen_images, chosen_labels


def proc_and_dump_sup_data(sub_set_data, split, sup_size=-1):
  images = sub_set_data["images"]
  labels = sub_set_data["labels"]
  if sup_size != -1:
    chosen_images, chosen_labels = get_data_by_size_lim(
        images, labels, sup_size)
  else:
    chosen_images = images
    chosen_labels = labels

  chosen_images = chosen_images / 255.0
  mean, std = augmentation_transforms.get_mean_and_std()
  chosen_images = (chosen_images - mean) / std
  example_list = []
  for image, label in zip(chosen_images, chosen_labels):
    # Write example to the tfrecord file
    example = tf.train.Example(features=tf.train.Features(
        feature={
            "image": _float_feature(image.reshape(-1)),
            "label": _int64_feature(label.reshape(-1))
        }))
    example_list += [example]
  out_path = os.path.join(
      FLAGS.output_base_dir,
      format_sup_filename(split, sup_size)
  )
  tf.logging.info(">> saving {} {} examples to {}".format(
      len(example_list), split, out_path))
  save_tfrecord(example_list, out_path)


def proc_and_dump_unsup_data(sub_set_data, aug_copy_num):
  ori_images = sub_set_data["images"].copy()

  image_idx = np.arange(len(ori_images))
  np.random.shuffle(image_idx)
  ori_images = ori_images[image_idx]

  # tf.logging.info("first 5 indexes after shuffling: {}".format(
  #     str(image_idx[:5])))

  ori_images = ori_images / 255.0
  mean, std = augmentation_transforms.get_mean_and_std()
  ori_images = (ori_images - mean) / std

  if FLAGS.task_name == "cifar10":
    aug_policies = found_policies.cifar10_policies()
  elif FLAGS.task_name == "svhn":
    aug_policies = found_policies.svhn_policies()

  example_list = []
  for image in ori_images:
    chosen_policy = aug_policies[np.random.choice(
        len(aug_policies))]
    aug_image = augmentation_transforms.apply_policy(
        chosen_policy, image)
    aug_image = augmentation_transforms.cutout_numpy(aug_image)

    # Write example to the tfrecord file
    example = tf.train.Example(features=tf.train.Features(
        feature={
            "ori_image": _float_feature(image.reshape(-1)),
            "aug_image": _float_feature(aug_image.reshape(-1)),
        }))
    example_list += [example]

  out_path = os.path.join(
      FLAGS.output_base_dir,
      format_unsup_filename(aug_copy_num),
  )
  save_tfrecord(example_list, out_path)


def main(unused_argv):

  output_base_dir = FLAGS.output_base_dir
  if not tf.gfile.Exists(output_base_dir):
    tf.gfile.MakeDirs(output_base_dir)

  data = load_dataset()
  if FLAGS.data_type == "sup":
    tf.logging.info("***** Processing supervised data *****")
    # process training set
    proc_and_dump_sup_data(data["train"], "train", sup_size=FLAGS.sup_size)
    # process test set
    proc_and_dump_sup_data(data["test"], "test")
  elif FLAGS.data_type == "unsup":
    tf.logging.info("***** Processing unsupervised data *****")
    # Just to make sure that different tfrecord files do not have data stored
    # in the same order. Since we read several tfrecord files in parallel, if
    # different tfrecord files have the same order, it is more probable that
    # multiple augmented examples of the same original example appear in the same
    # mini-batch.
    tf.logging.info(
        "using random seed {:d} for shuffling data".format(random_seed))
    np.random.seed(random_seed)
    for aug_copy_num in range(
        FLAGS.aug_copy_start, FLAGS.aug_copy_start + FLAGS.aug_copy):
      tf.logging.info(
          ">> processing aug copy # {}".format(aug_copy_num))
      proc_and_dump_unsup_data(data["train"], aug_copy_num)


if __name__ == "__main__":
  flags.DEFINE_enum(
      "task_name", "cifar10",
      enum_values=["cifar10", "svhn"], help="Task to use.")
  flags.DEFINE_enum(
      "data_type", "sup",
      enum_values=["sup", "unsup"],
      help="Whether to process supervised data or unsupervised data.")
  flags.DEFINE_string(
      "raw_data_dir", None, "Path of the raw data.")
  flags.DEFINE_string(
      "output_base_dir", "", "processed data path.")

  # configs for processing supervised data
  flags.DEFINE_bool(
      "use_equal_split", False, "If set to True, use equal number of data for each"
      "category. If set to False, use the same data as AutoAugment.")
  flags.DEFINE_integer(
      "sup_size", -1, "Number of supervised pairs to use."
      "-1: all training samples. 0: no supervised data.")

  # configs for processing unsupervised data
  flags.DEFINE_integer(
      "aug_copy", 0, "Number of augmented copies to create.")
  flags.DEFINE_integer(
      "aug_copy_start", 0, "The index of the first augmented copy.")

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
