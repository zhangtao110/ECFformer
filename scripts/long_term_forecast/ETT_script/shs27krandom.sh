export CUDA_VISIBLE_DEVICES=0

python -u gnn_train.py \
  --description test_string_bfs \
  --ppi_path ./data/protein.actions.SHS27k.STRING.txt \
  --pseq_path ./data/protein.SHS27k.sequences.dictionary.tsv \
  --vec_path ./data/vec5_CTC.txt \
  --split_new False \
  --split_mode random \
  --train_valid_index_path new_train_valid_index_json/RANDOM_SHS27k.json \
  --use_lr_scheduler True \
  --save_path ./save_model/ \
  --graph_only_train False \
  --batch_size 2048 \
  --epochs 300 \
  --model T1  

python -u gnn_train.py \
  --description test_string_bfs  \
  --ppi_path ./data/protein.actions.SHS27k.STRING.txt \
  --pseq_path ./data/protein.SHS27k.sequences.dictionary.tsv \
  --vec_path ./data/vec5_CTC.txt \
  --split_new False \
  --split_mode random \
  --train_valid_index_path new_train_valid_index_json/RANDOM_SHS27k.json \
  --use_lr_scheduler True \
  --save_path ./save_model/ \
  --graph_only_train False \
  --batch_size 2048 \
  --epochs 300 \
  --model T2


