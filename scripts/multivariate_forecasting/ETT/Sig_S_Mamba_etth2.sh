export CUDA_VISIBLE_DEVICES=0

model_name=feature_sig_fast
d state 2
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96_r \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --batch_size 16 \
  --dropout 0.1 \
  --d_model 256 \
  --use_progressive_signature  \
  --signature_depth 3 \
  --signature_fusion concat \
  --signature_window_size 8 \
  --d_ff 256 \
  --d_state 2 \
  --learning_rate 0.00002 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192_r \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 4 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --use_progressive_signature  \
  --signature_depth 3 \
  --signature_fusion concat \
  --signature_window_size 8 \
  --d_ff 256 \
  --d_state 32 \
  --learning_rate 0.00008 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2_r.csv \
  --model_id ETTh2_96_336_r \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 4 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --use_progressive_signature  \
  --signature_depth 3 \
  --signature_fusion concat \
  --signature_window_size 8 \
  --d_ff 256 \
  --d_state 32 \
  --learning_rate 0.00007 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 4 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --use_progressive_signature  \
  --signature_depth 3 \
  --signature_fusion concat \
  --signature_window_size 8 \
  --d_ff 256 \
  --d_state 32 \
  --learning_rate 0.00005 \
  --itr 1