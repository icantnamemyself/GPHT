device=0

# finetune
python run.py \
    --is_training 1 \
    --load_pretrain 1 \
    --transfer_data exchange_rate \
    --transfer_root_path dataset/exchange_rate/ \
    --transfer_data_path exchange_rate.csv \
    --data exchange_rate \
    --root_path dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --dropout 0.1 \
    --learning_rate 1e-5 \
    --gpu $device

# evalution
for pred in 96 192 336 720; do
  python run.py \
    --is_training 0 \
    --transfer_data exchange_rate \
    --transfer_root_path dataset/exchange_rate/ \
    --transfer_data_path exchange_rate.csv \
    --data exchange_rate \
    --root_path dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --ar_pred_len $pred \
    --learning_rate 1e-5 \
    --gpu $device
  done