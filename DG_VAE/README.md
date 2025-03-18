Using Auto-Encoder to enhance DeepGate2 structural embeddings

# Terminal

# 在运行 `torchrun` 命令之前，设置 PYTHONPATH
### 导入deepgate包
export PYTHONPATH=$PYTHONPATH:/home/xqgrp/wangjingxin/pythonproject/Multi-Gate-VAE/DG_VAE
### 配置参数
NUM_PROC=4
MODEL='DG_AE'
EXP_ID='DG_AE_NORM'
BATCH_SIZE=4
### train
torchrun --nproc_per_node=1 --master_port=29888 train.py --exp_id $EXP_ID --distributed --model $MODEL --batch_size 2 --num_epochs 300 --layernorm

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

