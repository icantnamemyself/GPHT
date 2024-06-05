device=0

# finetune
python run.py \
    --is_training 1 \
    --load_pretrain 1 \
    --transfer_data electricity \
    --transfer_root_path dataset/electricity/ \
    --transfer_data_path electricity.csv \
    --data electricity \
    --root_path dataset/electricity/ \
    --data_path electricity.csv \
    --dropout 0.1 \
    --learning_rate 0.0001 \
    --gpu $device

# evalution
for pred in 96 192 336; do
  python run.py \
    --is_training 0 \
    --transfer_data electricity \
    --transfer_root_path dataset/electricity/ \
    --transfer_data_path electricity.csv \
    --data electricity \
    --root_path dataset/electricity/ \
    --data_path electricity.csv \
    --ar_pred_len $pred \
    --gpu $device
  done