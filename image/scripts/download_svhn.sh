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
#!/bin/bash
task_name=svhn
mkdir -p data/proc_data/${task_name}
cd data/proc_data/${task_name}
aug_copy=$1

url_prefix=https://storage.googleapis.com/uda_model/image/proc_data_v1.1/${task_name}

wget $url_prefix/train-size_1000.tfrecord.0

for i in `seq 0 6`;
do
  wget $url_prefix/test.tfrecord.$i
done

# Using gsutil -m cp is much faster

aug_copy_end=$( expr $aug_copy - 1)
for i in `seq 0 $aug_copy_end`;
do
  for j in `seq 0 17`;
  do
    wget $url_prefix/unsup-$i.tfrecord.$j
  done
done
