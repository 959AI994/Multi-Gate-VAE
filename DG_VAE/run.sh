NUM_PROC=4
MODEL='DG_AE'
EXP_ID='DG_AE_NORM'
BATCH_SIZE=4

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py \
 --exp_id $EXP_ID \
 --distributed \
 --model $MODEL \
 --batch_size $BATCH_SIZE --num_epochs 300 \
 --layernorm
