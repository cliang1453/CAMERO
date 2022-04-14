#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "train.sh <gpu>"
  exit 1
fi
gpu=$1
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}

encoder_type=1 # model type, defined in pretrained_models.py
init_ckpt="/data/bert_model_base_uncased.pt" # path to the checkpoint for model initialization
train_datasets="mnli"
test_datasets="mnli_matched,mnli_mismatched"
data_dir="/data/glue/canonical_data/bert-base-uncased" # path to the directory containing preprocessed .json glue data

# training args
batch_size=32
batch_size_eval=32
epochs=3
lr="8e-5"

# camero args
teaching_type="pairwise"
pert_type="dropout"
n_models=4
kd_alpha=1

output_dir="/camero_output/${train_datasets}_${teaching_type}_${pert_type}_${n_models}_${kd_alpha}"
python collaborative_train.py \
--data_dir ${data_dir} --init_checkpoint ${init_ckpt} \
--batch_size ${batch_size} \
--batch_size_eval ${batch_size_eval} \
--output_dir ${output_dir} \
--log_file ${output_dir}/log.log \
--answer_opt 1 \
--optimizer "adamax" \
--train_datasets ${train_datasets} \
--test_datasets ${test_datasets} \
--grad_clipping 0 \
--global_grad_clipping 1 \
--epochs ${epochs} \
--encoder_type ${encoder_type} \
--learning_rate ${lr} \
--teaching_type ${teaching_type} --pert_type ${pert_type} --n_models ${n_models} --kd_alpha ${kd_alpha} --eval_ensemble \
--multi_gpu_on

