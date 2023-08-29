#!/bin/bash
#############################cifar10
##resnet20
#     resnet20 复现 0%
92.63%** CUDA_VISIBLE_DEVICES=0 python train.py --model resnet20 --BATCH_SIZE 512     --epochs 600  --LR 0.1 --dataset cifar10 --purity 1       --reclust_pur 1     --noisy_pro 0      --retention_ratio 1 --GB_mode
#     resnet20 复现 10%
92.09%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet20 --BATCH_SIZE 1024   --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.95  --reclust_pur 0.90 --noisy_pro 0.1 --retention_ratio 1 --GB_mode
#     resnet20 复现 20%
91.03%** CUDA_VISIBLE_DEVICES=0 python train.py --model resnet20 --BATCH_SIZE 1024   --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.90  --reclust_pur 0.85 --noisy_pro 0.2 --retention_ratio 1 --GB_mode
#     resnet20 复现 30%
89.43%** CUDA_VISIBLE_DEVICES=0 python train.py --model resnet20 --BATCH_SIZE 1024   --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.85  --reclust_pur 0.80 --noisy_pro 0.3 --retention_ratio 1 --GB_mode
#     resnet20 复现 40%
87.82%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet20 --BATCH_SIZE 1024   --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.80  --reclust_pur 0.75 --noisy_pro 0.4 --retention_ratio 1 --GB_mode
#     resnet20 复现 50%
85.93%** CUDA_VISIBLE_DEVICES=0 python train.py --model resnet20 --BATCH_SIZE 1024   --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.75  --reclust_pur 0.70 --noisy_pro 0.5 --retention_ratio 1 --GB_mode

# ############# no GB

# #resnet20 0% 10% 20% 30% 40% 50%
92.64%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet20 --BATCH_SIZE 128  --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0 
90.49%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet20 --BATCH_SIZE 128  --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.1   
88.79%** CUDA_VISIBLE_DEVICES=0 python train.py --model resnet20 --BATCH_SIZE 128  --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.2   
87.53%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet20 --BATCH_SIZE 128  --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.3   
85.42%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet20 --BATCH_SIZE 128  --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.4   
83.58%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet20 --BATCH_SIZE 128  --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.5   
