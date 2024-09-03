export CUDA_VISIBLE_DEVICES=0

model_name=Crossformer1_3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/YALI/ \
  --data_path yali1.csv \
  --model_id yali1_10_10 \
  --model $model_name \
  --data yali \
  --target press\
  --features M \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/YALI/ \
  --data_path yali1.csv \
  --model_id yali1_10_10 \
  --model $model_name \
  --data yali \
  --target press\
  --features M \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 72 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/YALI/ \
  --data_path yali1.csv \
  --model_id yali1_10_10 \
  --model $model_name \
  --data yali \
  --target press\
  --features M \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/YALI/ \
  --data_path yali1.csv \
  --model_id yali1_10_10 \
  --model $model_name \
  --data yali \
  --target press\
  --features M \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1