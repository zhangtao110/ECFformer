export CUDA_VISIBLE_DEVICES=0

model_name=Crossformer3_4

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/D1/ \
  --data_path yali.csv \
  --model_id yali_96_30 \
  --model $model_name \
  --data yali \
  --target press\
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --train_epochs 10 \
  --inverse \
  --use_dtw False \
  --batch_size 128 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/D1/ \
  --data_path yali.csv \
  --model_id yali_96_60 \
  --model $model_name \
  --data yali \
  --target press\
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 60 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --train_epochs 10 \
  --inverse \
  --batch_size 128 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/D1/ \
  --data_path yali.csv \
  --model_id yali_96_90 \
  --model $model_name \
  --data yali \
  --target press\
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 90 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --train_epochs 10 \
  --inverse \
  --batch_size 128 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/D1/ \
  --data_path yali.csv \
  --model_id yali_96_120 \
  --model $model_name \
  --data yali \
  --target press\
  --features MS \
  --seq_len 96 \
  --label_len 148 \
  --pred_len 120 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --train_epochs 10 \
  --inverse \
  --batch_size 128 \
  --itr 1
