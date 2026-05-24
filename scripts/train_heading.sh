#!/bin/sh
env="SingleControl"
scenario="1/heading"
algo="ppo"
exp="v1"
seed=5

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

# 【修改】去掉了 CUDA_VISIBLE_DEVICES=0，让 PyTorch 自动选择 CPU 运行
# 【修改】n-rollout-threads 也改回了 2。
# CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim.py \
python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 2 \
    --log-interval 1 --save-interval 1 \
    --num-mini-batch 5 --buffer-size 3000 --num-env-steps 1e8 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8


# 【运行】bash train_heading.sh