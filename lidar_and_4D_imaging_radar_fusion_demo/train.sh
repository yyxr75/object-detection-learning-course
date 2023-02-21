# training
CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup python -u -X faulthandler \
      -m torch.distributed.launch \
      --nproc_per_node=4 \
      train.py \
      --using_multi_scale 1 \
      --batch_size 16 \
      --start_epoch 0 \
      --end_epoch 200 \
      --num_workers 0 \
      --cuda 1 \
      --rotate_angle 0 \
      --fusion_arch preCat \
      --fusion_loss_arch fusion_loss_base\
      --binNum 8 \
      --showHeatmap 0 \
      --chooseLoss 0 \
      --lr_scheduler yoloxwarmcos \
      --base_lr 0.01 \
      --min_lr_ratio 0.05 \
      --warmup_epochs 5 \
      --warmup_lr 0 \
      --weight_decay 1e-4 \
      --main_gpuid 0 \
      --mAP_type pycoco \
      --obj_thresh 0.5 \
>> train_0616.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -X faulthandler -m torch.distributed.launch --nproc_per_node=4 train.py