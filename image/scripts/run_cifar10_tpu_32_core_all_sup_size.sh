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
model_dir=gs://qizhex/uda/image/ckpt/tpu_32_core  # change this

task_name=cifar10
data_dir=gs://uda_model/image/proc_data/${task_name}


# 4000 2000, 1000, 500
for sup_size in 4000 2000 1000 500;
do
  python main.py \
    --use_tpu=True \
    --do_train=True \
    --do_eval=False \
    --tpu=${train_tpu_name} \
    --task_name=${task_name} \
    --sup_size=${sup_size} \
    --unsup_ratio=30 \
    --tsa=linear_schedule \
    --ent_min_coeff=0.1 \
    --data_dir=${data_dir} \
    --model_dir=${model_dir} \
    $@
done

# 250
python main.py \
  --use_tpu=True \
  --do_train=True \
  --do_eval=False \
  --tpu=${train_tpu_name} \
  --task_name=${task_name} \
  --sup_size=250 \
  --unsup_ratio=40 \
  --tsa=log_schedule \
  --uda_confidence_thresh=0.8 \
  --uda_softmax_temp=0.9 \
  --train_steps=50000 \
  --warmup_steps=10000 \
  --weight_decay_rate=0.0007 \
  --unsup_coeff=6 \
  --ent_min_coeff=0.1 \
  --data_dir=${data_dir} \
  --model_dir=${model_dir} \
  $@
