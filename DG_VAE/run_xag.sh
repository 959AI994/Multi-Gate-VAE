NUM_PROC=2
MODEL='DG_AE'
EXP_ID='DG_AE_NORM_XAG'
BATCH_SIZE=16
TYPE=xag
EPOCH=60

# python train.py --exp_id $EXP_ID --model $MODEL --batch_size $BATCH_SIZE --num_epochs $EPOCH --layernorm  --type $TYPE
torchrun --nproc_per_node=2 --master_port=29766 \
    train.py \
    --exp_id $EXP_ID \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --num_epochs $EPOCH \
    --layernorm \
    --type $TYPE \
    --gpus 6,7 \
    --distributed