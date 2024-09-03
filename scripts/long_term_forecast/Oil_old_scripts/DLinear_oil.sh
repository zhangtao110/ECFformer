export CUDA_VISIBLE_DEVICES=0

model_name=DLinear

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Oil/ \
  --data_path pt104-2.csv \
  --model_id pt104_10_10 \
  --model $model_name \
  --data Oil \
  --target pressure\
  --features M \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --d_model 32 \
  --learning_rate 0.01


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Oil/ \
  --data_path pt104-2.csv \
  --model_id pt104_10_10 \
  --model $model_name \
  --data Oil \
  --target pressure\
  --features M \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --d_model 32 \
  --learning_rate 0.01


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Oil/ \
  --data_path pt104-2.csv \
  --model_id pt104_10_10 \
  --model $model_name \
  --data Oil \
  --target pressure\
  --features M \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 72 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --d_model 32 \
  --learning_rate 0.01


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Oil/ \
  --data_path pt104-2.csv \
  --model_id pt104_10_10 \
  --model $model_name \
  --data Oil \
  --target pressure\
  --features M \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --d_model 32 \
  --learning_rate 0.01
