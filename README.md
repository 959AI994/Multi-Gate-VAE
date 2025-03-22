# Resource
This repository's code is an extension of the DG-VAE (written by stone) version code (with the addition of multimodal and func loss).(Notice:Structure:GCN+AE;Function:GNN)
### DG-VAE
location: https://github.com/zshi0616/DG_VAE （Using GCN+AE to update structure）
### Deepgate
location: https://github.com/zshi0616/python-deepgate (Using GNN to update structure and function)

# Terminal

### 在运行 `torchrun` 命令之前，设置 PYTHONPATH,导入deepgate包
export PYTHONPATH=$PYTHONPATH:/home/xqgrp/wangjingxin/pythonproject/Multi-Gate-VAE/DG_VAE
### 配置参数
NUM_PROC=4
MODEL='DG_AE'
EXP_ID='DG_AE_NORM'
BATCH_SIZE=4
### debug
python train.py --exp_id $EXP_ID --model $MODEL --batch_size 2 --num_epochs 300 --layernorm
### train
torchrun --nproc_per_node=1 --master_port=29888 train.py --exp_id $EXP_ID --distributed --model $MODEL --batch_size 2 --num_epochs 300 --layernorm <br>

torchrun \
--nproc_per_node=$NUM_PROC \
--master_port=29888 \
train.py \
--exp_id $EXP_ID \
--distributed \
--model $MODEL \
--batch_size $BATCH_SIZE \
--num_epochs 300 \
--layernorm

# Introduction
We can use the code in this repository to extract embeddings of different modes, such as xag, mig, xmg, aig. Using Auto-Encoder to enhance DeepGate2 structural embeddings.

# Notice
bench_parser类中，训练xmg,mig,xag的时候需要将edge_index进行转置，“edge_index = edge_index.t().contiguous()”，aig则不需要转置，把该行代码注释掉即可<br>
共六个聚合器，需要考虑6种，gate_to_index = {'INPUT': 0, 'MAJ': 1, 'NOT': 2, 'AND': 3, 'OR': 4, 'XOR': 5} <br>