export CUDA_VISIBLE_DEVICES=0

model_name=feature_sig
d_state = 32
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/PEMS/ \
#   --data_path PEMS04.npz \
#   --model_id PEMS04_96_12 \
#   --model $model_name \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --pred_len 12 \
#   --e_layers 4 \
#   --enc_in 358 \
#   --dec_in 358 \
#   --c_out 358 \
#   --dropout 0.1 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --use_progressive_signature  \
#   --signature_depth 3 \
#   --signature_fusion concat \
#   --signature_window_size 8 \
#   --learning_rate 0.0005 \
#   --train_epochs 15 \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/PEMS/ \
#   --data_path PEMS04.npz \
#   --model_id PEMS04_96_24 \
#   --model $model_name \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --pred_len 24 \
#   --e_layers 4 \
#   --enc_in 358 \
#   --dec_in 358 \
#   --c_out 358 \
#   --des 'Exp' \
#   --d_model 512 \
#   --use_progressive_signature  \
#   --signature_depth 3 \
#   --signature_fusion concat \
#   --signature_window_size 8 \
#   --d_ff 512 \
#   --learning_rate 0.0003 \
#   --train_epochs 15 \
#   --itr 1


# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/PEMS/ \
#   --data_path PEMS04.npz \
#   --model_id PEMS04_96_48 \
#   --model $model_name \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --pred_len 48 \
#   --e_layers 5 \
#   --enc_in 358 \
#   --dec_in 358 \
#   --c_out 358 \
#   --des 'Exp' \
#   --d_model 512 \
#   --use_progressive_signature  \
#   --signature_depth 3 \
#   --signature_fusion concat \
#   --signature_window_size 8 \
#   --d_ff 512 \
#   --learning_rate 0.0003 \
#   --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.npz \
  --model_id PEMS04_96_96 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --d_model 512 \
  --use_progressive_signature  \
  --signature_depth 3 \
  --signature_fusion concat \
  --signature_window_size 8 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --itr 1