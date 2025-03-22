NUM_PROC=2
MODEL='DG_AE'
EXP_ID='DG_AE_NORM_MIG'
BATCH_SIZE=16
TYPE=mig
EPOCH=60
GPU=4,5
# python train.py --exp_id $EXP_ID --model $MODEL --batch_size $BATCH_SIZE --num_epochs $EPOCH --layernorm  --type $TYPE
# torchrun --nproc_per_node=2 --master_port=29888 --exp_id $EXP_ID --model $MODEL --batch_size $BATCH_SIZE --num_epochs $EPOCH --layernorm  --type $TYPE  --gpus 4,5

torchrun --nproc_per_node=2 --master_port=29755 \
    train.py \
    --exp_id $EXP_ID \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --num_epochs $EPOCH \
    --layernorm \
    --type $TYPE \
    --gpus $GPU \
    --distributed