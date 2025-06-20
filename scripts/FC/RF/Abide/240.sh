#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

mkdir -p ./logs/Abide_240/RF
log_dir="./logs/Abide_240/RF/"

model_name=DeCI
seeds=(2024)
bss=(16)
lrs=(1e-3)
layers=(2)
dropouts=(0.)
d_models=(64)

for seed in "${seeds[@]}"; do
    for bs in "${bss[@]}"; do
        for lr in "${lrs[@]}"; do
            for layer in "${layers[@]}"; do
                for dropout in "${dropouts[@]}"; do
                    for d_model in "${d_models[@]}"; do
                                    python -u run_cv.py \
                                    --model $model_name \
                                    --Method RF \
                                    --data_path ./dataset/Abide \
                                    --data Abide \
                                    --protocol AAL116\
                                    --channel 116 \
                                    --seq_len 240\
                                    --classes 2\
                                    --seed $seed \
                                    --batch_size $bs \
                                    --learning_rate $lr \
                                    --layer $layer\
                                    --dropout $dropout\
                                    --d_model $d_model\
                                    --loss mse\
                                    --use_norm 1 >"${log_dir}sd${seed}_bs${bs}_lr${lr}_ly${layer}_dp${dropout}_dm${d_model}.log"
                    done
                done
            done
        done
    done
done
