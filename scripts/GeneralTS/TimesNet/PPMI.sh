#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

mkdir -p ./logs/PPMI/TimesNet
log_dir="./logs/PPMI/TimesNet/"

model_name=TimesNet
seeds=(2024)
bss=(16)
lrs=(1e-3)
layers=(1 2)
dropouts=(0. 0.2)
d_models=(64 128)

for seed in "${seeds[@]}"; do
    for bs in "${bss[@]}"; do
        for lr in "${lrs[@]}"; do
            for layer in "${layers[@]}"; do
                for dropout in "${dropouts[@]}"; do
                    for d_model in "${d_models[@]}"; do
                                    python -u run_cv.py \
                                    --model $model_name \
                                    --data_path ./dataset/PPMI \
                                    --data PPMI \
                                    --protocol AAL116\
                                    --channel 116 \
                                    --seq_len 210\
                                    --classes 4\
                                    --seed $seed \
                                    --batch_size $bs \
                                    --learning_rate $lr \
                                    --layer $layer\
                                    --dropout $dropout\
                                    --d_model $d_model\
                                    --loss ce\
                                    --use_norm 1 >"${log_dir}sd${seed}_bs${bs}_lr${lr}_ly${layer}_dp${dropout}_dm${d_model}.log"
                    done
                done
            done
        done
    done
done
