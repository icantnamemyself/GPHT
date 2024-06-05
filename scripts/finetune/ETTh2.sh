device=0

# finetune
python run.py \
    --is_training 1 \
    --load_pretrain 1 \
    --transfer_data ETTh2 \
    --transfer_root_path dataset/ETT-small/ \
    --transfer_data_path ETTh2.csv \
    --data ETTh2 \
    --root_path dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --dropout 0.0 \
    --learning_rate 0.0001 \
    --gpu $device

# evalution
for pred in 96 192 336 720; do
  python run.py \
    --is_training 0 \
    --transfer_data ETTh2 \
    --transfer_root_path dataset/ETT-small/ \
    --transfer_data_path ETTh2.csv \
    --data ETTh2 \
    --root_path dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --ar_pred_len $pred \
    --gpu $device
  done