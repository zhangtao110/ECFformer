export CUDA_VISIBLE_DEVICES=0

model_name=FEDformer

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
  --learning_rate 0.001 \
  --train_epochs 30 \
  --lradj type3 \
  --patience 100

