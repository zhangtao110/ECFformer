export CUDA_VISIBLE_DEVICES=0

model_name=DLinear_Transformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/pumping/ \
  --data_path cc8-244_5.csv \
  --model_id cc8 \
  --model $model_name \
  --data Oil \
  --target pressure\
  --features M \
  --seq_len 96 \
  --label_len 48 \
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
  --d_model 32

