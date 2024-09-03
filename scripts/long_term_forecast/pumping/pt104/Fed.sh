export CUDA_VISIBLE_DEVICES=0

model_name=Fedformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/pumping_remove/ \
  --data_path pt104.csv \
  --model_id pt104 \
  --model $model_name \
  --data Oil \
  --target pressure\
  --features M \
  --seq_len 60 \
  --label_len 30 \
  --pred_len 60 \
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
  --train_epochs 20 \
  --lradj type3 \
  --patience 5

