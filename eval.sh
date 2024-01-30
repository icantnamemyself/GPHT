device=1

for data in ETTh1 ETTh2 ETTm1 ETTm2; do
  for pred in 96 192 336 720; do
    echo $data
    python run.py \
      --is_training 0 \
      --transfer_data $data \
      --transfer_root_path dataset/ETT-small/ \
      --transfer_data_path $data.csv \
      --ar_pred_len $pred \
      --gpu $device
  done
done

for data in weather exchange_rate electricity traffic; do
  for pred in 96 192 336 720; do
    echo $data
    python run.py \
      --is_training 0 \
      --transfer_data $data \
      --transfer_root_path dataset/$data/ \
      --transfer_data_path $data.csv \
      --ar_pred_len $pred \
      --gpu $device
  done
done
