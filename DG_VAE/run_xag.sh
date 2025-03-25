NUM_PROC=2
MODEL='DG_AE'
EXP_ID='DG_AE_NORM_XAG'
BATCH_SIZE=16
TYPE=xag
EPOCH=60

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29655 \
    train.py \
    --exp_id $EXP_ID \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --num_epochs $EPOCH \
    --layernorm \
    --type $TYPE \
    --distributed