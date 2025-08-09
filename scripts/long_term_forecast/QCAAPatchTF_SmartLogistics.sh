model_name=QCAAPatchTF

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/New_test/ \
  --data_path smart_logistics_prepared.csv \
  --model_id SmartLogistics_96_96 \
  --model QCAAPatchTF \
  --channel_independence 0 \
  --data SmartLogistics \
  --features M \
  --freq d \
  --target Inventory_Level \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'SmartLogisticsExp' \
  --itr 1
