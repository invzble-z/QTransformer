model_name=QCAAPatchTF

# Người mua
for level in "Customer_Country" "Customer_Country_Customer_State" "Customer_Country_Customer_State_Customer_City"
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path buyer_${level}_processed.csv \
      --model_id SupplyChain_${level} \
      --model $model_name \
      --channel_independence 0 \
      --data SupplyChain \
      --features MS \
      --seq_len 90 \
      --label_len 45 \
      --pred_len 7 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 25 \
      --dec_in 25 \
      --c_out 1 \
      --des 'Exp' \
      --itr 1
done

# # Người bán
# for level in "Order_Country" "Order_Country_Order_State" "Order_Country_Order_State_Order_City"
# do
#     python -u run.py \
#       --task_name long_term_forecast \
#       --is_training 1 \
#       --root_path ./dataset/ \
#       --data_path seller_${level}_processed.csv \
#       --model_id SupplyChain_${level} \
#       --model $model_name \
#       --channel_independence 0 \
#       --data SupplyChain \
#       --features MS \
#       --seq_len 90 \
#       --label_len 45 \
#       --pred_len 7 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 27 \
#       --dec_in 27 \
#       --c_out 1 \
#       --des 'Exp' \
#       --itr 1
# done