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

train_tpu_name=node-1 # change this
eval_tpu_name=node-2 # change this
model_dir=gs://qizhex/uda/image/ckpt/svhn_tpu_32_core  # change this

task_name=svhn
data_dir=gs://uda_model/image/proc_data/${task_name}

# training
python main.py \
  --use_tpu=True \
  --do_train=True \
  --do_eval=False \
  --tpu=${train_tpu_name} \
  --task_name=${task_name} \
  --sup_size=1000 \
  --aug_copy=100 \
  --unsup_ratio=40 \
  --weight_decay_rate=7e-4 \
  --tsa=linear_schedule \
  --data_dir=${data_dir} \
  --model_dir=${model_dir} \
  --learning_rate=0.05 \
  $@

# eval
python main.py \
  --use_tpu=True \
  --do_train=False \
  --do_eval=True \
  --tpu=${eval_tpu_name} \
  --task_name=${task_name} \
  --data_dir=${data_dir} \
  --model_dir=${model_dir}
