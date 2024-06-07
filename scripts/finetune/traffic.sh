device=0

 finetune
python run.py \
    --is_training 1 \
    --load_pretrain 1 \
    --transfer_data traffic \
    --transfer_root_path dataset/traffic/ \
    --transfer_data_path traffic.csv \
    --data traffic \
    --root_path dataset/traffic/ \
    --data_path traffic.csv \
    --dropout 0.0 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --gpu $device

# evalution
for pred in 96 192 336 720; do
  python run.py \
    --is_training 0 \
    --transfer_data traffic \
    --transfer_root_path dataset/traffic/ \
    --transfer_data_path traffic.csv \
    --data traffic \
    --root_path dataset/traffic/ \
    --data_path traffic.csv \
    --ar_pred_len $pred \
    --learning_rate 5e-5 \
    --gpu $device
  done