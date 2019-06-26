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

python main.py \
  --use_tpu=False \
  --do_train=True \
  --do_eval=True \
  --task_name=${task_name} \
  --sup_size=4000 \
  --unsup_ratio=5 \
  --train_batch_size=64 \
  --data_dir=data/proc_data/${task_name} \
  --model_dir=ckpt/cifar10_gpu \
  --train_steps=400000 \
  $@
