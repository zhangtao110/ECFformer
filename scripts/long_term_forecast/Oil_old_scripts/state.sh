export CUDA_VISIBLE_DEVICES=0

model_name=ARIMA

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Oil/ \
  --data_path pt104.csv \
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
  --enc_in 12 \
  --dec_in 12 \
  --c_out 12 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --d_model 32 \
  --learning_rate 0.01
