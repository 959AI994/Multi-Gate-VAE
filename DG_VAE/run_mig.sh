NUM_PROC=4
MODEL='DG_AE'
EXP_ID='DG_AE_NORM'
BATCH_SIZE=4
TYPE=mig
EPOCH=100

python train.py --exp_id $EXP_ID --model $MODEL --batch_size $BATCH_SIZE --num_epochs $EPOCH --layernorm  --type $TYPE