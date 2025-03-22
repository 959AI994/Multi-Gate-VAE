# Introduction
This repository's code is an extension of the DG-VAE version code (with the addition of multimodal and func loss).We can use the code in this repository to extract embeddings of different modes, such as xag, mig, xmg, aig.

### Deepgate
location: https://github.com/zshi0616/python-deepgate (Using GNN to update structure and function)
### DG-VAE
location: https://github.com/zshi0616/DG-VAE

# Terminal

### 在运行 sh脚本命令之前，设置 PYTHONPATH,导入deepgate包
export PYTHONPATH=$PYTHONPATH:/home/xqgrp/wangjingxin/pythonproject/Multi-Gate-VAE/DG_VAE
### 训练mig的encoder
bash run_mig.sh
### 训练aig的encoder
bash run_aig.sh
### 训练xmg的encoder
bash run_xmg.sh
### 训练xag的encoder
bash run_xag.sh

# 若不使用sh，直接在终端运行
### 配置参数
NUM_PROC=4
MODEL='DG_AE'
EXP_ID='DG_AE_NORM'
BATCH_SIZE=4
### debug
python train.py --exp_id $EXP_ID --model $MODEL --batch_size 64 --num_epochs 100 --layernorm  --type mig
### 分布train
torchrun --nproc_per_node=2 --master_port=29888 train.py --exp_id $EXP_ID --distributed --model $MODEL --batch_size 64 --num_epochs 100 --layernorm   --gpus 4,5  --type mig 

# Notice
共六个聚合器，需要考虑6种，gate_to_index = {'INPUT': 0, 'MAJ': 1, 'NOT': 2, 'AND': 3, 'OR': 4, 'XOR': 5} 