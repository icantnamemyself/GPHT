device=0

python run.py \
    --is_training 1 \
    --data pretrain \
    --GT_d_model 512 \
    --GT_d_ff 2048 \
    --token_len 48 \
    --GT_pooling_rate [8,4,2,1] \
    --GT_e_layers 3 \
    --depth 4 \
    --learning_rate 0.0001 \
    --gpu $device
