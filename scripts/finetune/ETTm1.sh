device=0

# finetune
python run.py \
    --is_training 1 \
    --load_pretrain 1 \
    --transfer_data ETTm1 \
    --transfer_root_path dataset/ETT-small/ \
    --transfer_data_path ETTm1.csv \
    --data ETTm1 \
    --root_path dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --dropout 0.0 \
    --learning_rate 0.001 \
    --gpu $device

# evalution
for pred in 96 192 336 720; do
  python run.py \
    --is_training 0 \
    --transfer_data ETTm1 \
    --transfer_root_path dataset/ETT-small/ \
    --transfer_data_path ETTm1.csv \
    --data ETTm1 \
    --root_path dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --ar_pred_len $pred \
    --learning_rate 0.001 \
    --gpu $device
  done