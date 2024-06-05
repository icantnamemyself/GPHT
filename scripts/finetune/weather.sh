device=0

# finetune
python run.py \
    --is_training 1 \
    --load_pretrain 1 \
    --transfer_data weather \
    --transfer_root_path dataset/weather/ \
    --transfer_data_path weather.csv \
    --data weather \
    --root_path dataset/weather/ \
    --data_path weather.csv \
    --dropout 0.1 \
    --learning_rate 0.0001 \
    --gpu $device

# evalution
for pred in 96 192 336 720; do
  python run.py \
    --is_training 0 \
    --transfer_data weather \
    --transfer_root_path dataset/weather/ \
    --transfer_data_path weather.csv \
    --data weather \
    --root_path dataset/weather/ \
    --data_path weather.csv \
    --ar_pred_len $pred \
    --gpu $device
  done