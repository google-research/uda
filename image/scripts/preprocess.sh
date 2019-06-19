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

task_name=cifar10

# preprocess supervised data
python preprocess.py \
  --data_type=sup \
  --sup_size=4000 \
  --task_name=${task_name} \
  --raw_data_dir=data/raw_data/${task_name} \
  --output_base_dir=data/proc_data/${task_name}

# preprocess unsupervised data
python preprocess.py \
  --data_type=unsup \
  --task_name=${task_name} \
  --raw_data_dir=data/raw_data/${task_name} \
  --output_base_dir=data/proc_data/${task_name} \
  $@

