NUM_PROC=3
MODEL='DG_AE'
EXP_ID='DG_AE_NORM_AIG'
BATCH_SIZE=16
TYPE=aig
EPOCH=60

# python train.py --exp_id $EXP_ID --model $MODEL --batch_size $BATCH_SIZE --num_epochs $EPOCH --layernorm  --type $TYPE
# torchrun --nproc_per_node=2 --master_port=29855 \
#     train.py \
#     --exp_id $EXP_ID \
#     --model $MODEL \
#     --batch_size $BATCH_SIZE \
#     --num_epochs $EPOCH \
#     --layernorm \
#     --type $TYPE \
#     --gpus 4,5 \
#     --distributed 
# 修改后：
CUDA_VISIBLE_DEVICES=3,4,5 torchrun --nproc_per_node=3 --master_port=29855 \
    train.py \
    --exp_id $EXP_ID \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --num_epochs $EPOCH \
    --layernorm \
    --type $TYPE \
    --distributed
