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
bert_vocab_file=pretrained_models/bert_base/vocab.txt


# Preprocess supervised training set
python preprocess.py \
  --raw_data_dir=data/IMDB_raw/csv \
  --output_base_dir=data/proc_data/IMDB/train_20 \
  --data_type=sup \
  --sub_set=train \
  --sup_size=20 \
  --vocab_file=$bert_vocab_file \
  $@

# Preprocess test set
python preprocess.py \
  --raw_data_dir=data/IMDB_raw/csv \
  --output_base_dir=data/proc_data/IMDB/dev \
  --data_type=sup \
  --sub_set=dev \
  --vocab_file=$bert_vocab_file \
  $@


# Preprocess unlabeled set
python preprocess.py \
  --raw_data_dir=data/IMDB_raw/csv \
  --output_base_dir=data/proc_data/IMDB/unsup \
  --back_translation_dir=data/back_translation/imdb_back_trans \
  --data_type=unsup \
  --sub_set=unsup_in \
  --aug_ops=bt-0.9 \
  --aug_copy_num=0 \
  --vocab_file=$bert_vocab_file \
  $@
