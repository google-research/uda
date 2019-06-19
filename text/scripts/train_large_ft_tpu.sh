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
train_tpu=node-1
eval_tpu=node-2
model_dir=gs://qizhex/uda/text/ckpt/large_ft_exp_1

python main.py \
  --use_tpu=True \
  --tpu_name=${train_tpu} \
  --do_train=True \
  --do_eval=False \
  --sup_train_data_dir= \
  --eval_data_dir= \
  --bert_config_file= \
  --vocab_file= \
  --init_checkpoint= \
  --task_name=IMDB \
  --model_dir=gs://uda_data/UDA/ckpt/large_ft \
  --max_seq_length=512 \
  --num_train_steps=3000 \
  --learning_rate=3e-05 \
  --train_batch_size=32 \
  --num_warmup_steps=300

python main.py \
  --use_tpu=True \
  --tpu_name=${eval_tpu} \
  --do_train=False \
  --do_eval=True \
  --sup_train_data_dir= \
  --eval_data_dir= \
  --bert_config_file= \
  --vocab_file= \
  --task_name=IMDB \
  --model_dir=${model_dir} \
  --max_seq_length=512 \
  --eval_batch_size=8 \
