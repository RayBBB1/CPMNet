python train.py ^
--train D:\Bme_Dataset\ME\ME_db1_20241210\data_split\train.txt ^
--val_set D:\Bme_Dataset\ME\ME_db1_20241210\data_split\val.txt ^
--test_set D:\Bme_Dataset\ME\ME_db1_20241210\data_split\test.txt ^
--batch 6 ^
--num_samples 5  ^
--mixed_precision  ^
--exp_name ME_db1 -^
-val_mixed_precision ^
--epochs 300 ^
--start_val_epoch 150 ^
--cls_num_neg -1  ^
--ema_warmup_epochs 120 ^
--iters_to_accumulate 2 ^
--lr 0.002 ^
--decay_gamma 0.1 ^
--pos_ignore_ratio 5 ^
--tp_ratio 0.6 ^
--use_itk_rotate ^
--lambda_offset 1.0 ^
--nodule_size_mode dhw
