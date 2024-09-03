export CUDA_VISIBLE_DEVICES=1

model_name=ARIMA

python -u run_stat.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/D3/ \
  --data_path yali.csv \
  --model_id yali_300_30 \
  --model $model_name \
  --data yali \
  --target press\
  --features MS \
  --seq_len 300 \
  --label_len 150 \
  --pred_len 30 \
  --des 'Exp' \
  --inverse \
  --itr 1

python -u run_stat.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/D3/ \
  --data_path yali.csv \
  --model_id yali_300_60 \
  --model $model_name \
  --data yali \
  --target press\
  --features MS \
  --seq_len 300 \
  --label_len 150 \
  --pred_len 60 \
  --des 'Exp' \
  --inverse \
  --itr 1

python -u run_stat.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/D3/ \
  --data_path yali.csv \
  --model_id yali_300_90 \
  --model $model_name \
  --data yali \
  --target press\
  --features MS \
  --seq_len 300 \
  --label_len 150 \
  --pred_len 90 \
  --des 'Exp' \
  --inverse \
  --itr 1

python -u run_stat.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/D3/ \
  --data_path yali.csv \
  --model_id yali_300_120 \
  --model $model_name \
  --data yali \
  --target press\
  --features MS \
  --seq_len 300 \
  --label_len 150 \
  --pred_len 120 \
  --des 'Exp' \
  --inverse \
  --itr 1