# python train.py --train_set ./data/pretrained_train.txt --val_set ./data/pretrained_val.txt --test_set ./data/all_client_test.txt --batch_size 3 --num_samples 5 --mixed_precision --val_mixed_precision
# python train.py --train_set ./data/test_bug.txt --val_set ./data/test_bug.txt --test_set ./data/test_bug.txt --batch_size 6 --num_samples 5 --mixed_precision --val_mixed_precision --crop_size 96 96 96 --first_stride 2 2 2 --pos_target_topk 7
python train.py --train_set /root/notebooks/automl/CPMNet/data_split/db1_1/train.txt --val_set /root/notebooks/automl/CPMNet/data_split/db1_1/val.txt --test_set /root/notebooks/automl/CPMNet/data_split/db1_1/test.txt --batch 8 --num_samples 4 --mixed_precision --exp_name ME_db1 --val_mixed_precision --epochs 300 --start_val_epoch 80 --cls_num_neg -1  --ema_warmup_epochs 80 --iters_to_accumulate 2 --lr 0.0003 --decay_gamma 0.3 --pos_ignore_ratio 5 --tp_ratio 0.6 --use_itk_rotate --nodule_size_mode dhw  