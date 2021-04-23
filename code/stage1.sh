#!/usr/bin/bash
# train stage1 model with train data
python3 train.py \
--version 0 \
--seed 42 \
--data_root ../dataset \
--num_workers 4 \
--batch_size 18 \
--model baseline \
--snapshot_size 50 \
--max_lr 1e-3 \
--min_lr 7e-5 \
--momentum 0.9 \
--weight_decay 1e-4 \
--val_fold_idx 3 \
--max_epoch 300 \
--gpus 1 \
--progress_bar_refresh_rate 20 \
--num_sanity_val_steps 2

--version 1
--log_dir ../logs/stage1
--seed 42

--data_root ../dataset
--add_depth t
--resize_pad t
--num_workers 3
--batch_size 16

--model phalanx_res34v4

--snapshot_size 50
--max_lr 1.2e-2
--min_lr 1e-3
--momentum 0.9
--weight_decay 1e-4
--val_fold_idx 0

--max_epoch 100
--gpus 1
--progress_bar_refresh_rate 20
--num_sanity_val_steps 2