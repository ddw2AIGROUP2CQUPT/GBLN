# #!/bin/bash
# #############################cifar10
# ##resnet20
# # 23/5/7      resnet20 复现 0%
92.63%** CUDA_VISIBLE_DEVICES=0 python train.py --model preactresnet18 --BATCH_SIZE 512    --epochs 300 --LR 0.02 --dataset cifar10 --purity 1     --reclust_pur 1     --noisy_pro 0   --retention_ratio 1 --GB_mode

# # 23/5/7      resnet20 复现 10%
# # 92.09%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet20 --BATCH_SIZE 1024   --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.85  --reclust_pur 0.85  --noisy_pro 0.1 --retention_ratio 1 --GB_mode

# #23/5/7       resnet20 复现 20%
# 92.07%** CUDA_VISIBLE_DEVICES=0 python train.py --model resnet20 --BATCH_SIZE 1024   --epochs 600 --LR 0.1 --dataset cifar10 --purity 0.90   --reclust_pur 0.85 --noisy_pro 0.2 --retention_ratio 1 --GB_mode
CUDA_VISIBLE_DEVICES=1 python train.py --model preactresnet18 --BATCH_SIZE 1024   --epochs 600 --LR 0.02 --dataset cifar10 --purity 0.90   --reclust_pur 0.75 --noisy_pro 0.2 --retention_ratio 1 --GB_mode
CUDA_VISIBLE_DEVICES=0 python train.py --model preactresnet18 --BATCH_SIZE 512   --epochs 400 --LR 0.01 --dataset cifar10 --purity 0.8   --reclust_pur 0.6 --noisy_pro 0.2 --retention_ratio 1 --GB_mode

# #23/5/7       resnet20 复现 30%
# # 89.43%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet20 --BATCH_SIZE 1024 --LR 0.1 --dataset cifar10 --purity 0.85 --reclust_pur 0.8 --noisy_pro 0.3 --retention_ratio 1 --GB_mode

# #23/5/7       resnet20 复现 40%
# 90.18%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet20 --BATCH_SIZE 1024 --epochs 600 --LR 0.1 --dataset cifar10 --purity 0.8 --reclust_pur 0.8 --noisy_pro 0.4 --retention_ratio 1 --GB_mode
# CUDA_VISIBLE_DEVICES=1 python train.py --model resnet20 --BATCH_SIZE 1024 --epochs 600 --LR 0.1 --dataset cifar10 --purity 0.8 --reclust_pur 0.65 --noisy_pro 0.4 --retention_ratio 1 --GB_mode

# #23/5/9       resnet20 复现 50%
# # 85.93%** CUDA_VISIBLE_DEVICES=0 python train.py --model resnet20 --BATCH_SIZE 1024 --LR 0.1 --dataset cifar10 --purity 0.75 --reclust_pur 0.7 --noisy_pro 0.5 --retention_ratio 1 --GB_mode


# ##densenet121
# # 23/5/7     densenet121 复现 0%
# # 94.17%** CUDA_VISIBLE_DEVICES= 0 python train.py --model densenet121 --BATCH_SIZE 512 --epochs 600 --LR 0.01 --dataset cifar10 --purity 0.95 --reclust_pur 0.95 --noisy_pro 0 --retention_ratio 1  --GB_mode

# # 23/5/7     densenet121 复现 10%
# # 92.67%** CUDA_VISIBLE_DEVICES=1 python train.py --model densenet121 --BATCH_SIZE 512 --epochs 600 --LR 0.01 --dataset cifar10 --purity 0.9 --reclust_pur 0.6 --noisy_pro 0.1 --retention_ratio 1  --GB_mode

# # 23/5/8     densenet121 复现 20% 
# 90.92%** CUDA_VISIBLE_DEVICES=0 python train.py --model densenet121 --BATCH_SIZE 512 --epochs 300 --LR 0.01 --dataset cifar10 --purity 0.85 --reclust_pur 0.65 --noisy_pro 0.2 --retention_ratio 1  --GB_mode

# # 23/5/8     densenet121 复现 30%
# # 88.72%** CUDA_VISIBLE_DEVICES=1 python train.py --model densenet121 --BATCH_SIZE 512 --epochs 600 --LR 0.01 --dataset cifar10 --purity 0.80 --reclust_pur 0.65 --noisy_pro 0.3 --retention_ratio 1  --GB_mode

# # 23/5/10     densenet121 复现 40%
# 85.35%** CUDA_VISIBLE_DEVICES=1 python train.py --model densenet121 --BATCH_SIZE 512 --epochs 300 --LR 0.1 --dataset cifar10 --purity 0.75 --reclust_pur 0.6 --noisy_pro 0.4 --retention_ratio 1  --GB_mode

# # 23/5/10     densenet121 复现 50%
# # 80.13%** CUDA_VISIBLE_DEVICES=0 python train.py --model densenet121 --BATCH_SIZE 512 --epochs 600 --LR 0.01 --dataset cifar10 --purity 0.7 --reclust_pur 0.6 --noisy_pro 0.5 --retention_ratio 1  --GB_mode

# ##resnet32
# # 23/5/10     resnet32 10%
# # 92.3%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet32 --BATCH_SIZE 1024 --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.95 --reclust_pur 0.65 --noisy_pro 0.1 --retention_ratio 1  --GB_mode

# ##resnet44
# # 23/5/10    .70.164  resnet44 0%
# 93.55%** CUDA_VISIBLE_DEVICES=0 python train.py --model resnet44 --BATCH_SIZE 512 --epochs 600 --LR 0.1 --dataset cifar10 --purity 1 --reclust_pur 1 --noisy_pro 0 --retention_ratio 1  --GB_mode
# # 23/5/10    .127.75  resnet44  10%
# 92.5%**  CUDA_VISIBLE_DEVICES=1 python train.py --model resnet44 --BATCH_SIZE 1024 --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.95 --reclust_pur 0.8 --noisy_pro 0.1 --retention_ratio 1  --GB_mode
# # 23/5/10    .70.164  resnet44  20%  
# 91.5%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet44 --BATCH_SIZE 1024 --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.9 --reclust_pur 0.75 --noisy_pro 0.2 --retention_ratio 1  --GB_mode
# # 23/5/10    .127.75  resnet44  30% 
# 89.89%** CUDA_VISIBLE_DEVICES=0 python train.py --model resnet44 --BATCH_SIZE 1024 --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.85 --reclust_pur 0.7 --noisy_pro 0.3 --retention_ratio 1  --GB_mode
# # 23/5/10    .127.75  resnet44  40% 
# 88.23%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet44 --BATCH_SIZE 1024 --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.8 --reclust_pur 0.65 --noisy_pro 0.4 --retention_ratio 1  --GB_mode
# # 23/5/10    .36.99  resnet44  50% 
# 85.64%** CUDA_VISIBLE_DEVICES=0 python train.py --model resnet44 --BATCH_SIZE 1024 --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.75 --reclust_pur 0.6 --noisy_pro 0.5 --retention_ratio 1  --GB_mode


# ##resnet56
# # 23/5/22    .127.75  resnet44 0%
# 94.10% CUDA_VISIBLE_DEVICES=0 python train.py --model resnet56 --BATCH_SIZE 512 --epochs 600 --LR 0.1 --dataset cifar10 --purity 1 --reclust_pur 1 --noisy_pro 0 --retention_ratio 1  --GB_mode
# # 23/5/22    .127.75  resnet44  10%
# 92.90% CUDA_VISIBLE_DEVICES=1 python train.py --model resnet56 --BATCH_SIZE 1024 --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.95 --reclust_pur 0.8 --noisy_pro 0.1 --retention_ratio 1  --GB_mode
# # 23/5/22    .42.178  resnet44  20%  
# 91.16% CUDA_VISIBLE_DEVICES=0 python train.py --model resnet56 --BATCH_SIZE 1024 --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.9 --reclust_pur 0.75 --noisy_pro 0.2 --retention_ratio 1  --GB_mode
# # 23/5/22    .42.178  resnet44  30% 
# 89.96% CUDA_VISIBLE_DEVICES=1 python train.py --model resnet56 --BATCH_SIZE 1024 --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.85 --reclust_pur 0.7 --noisy_pro 0.3 --retention_ratio 1  --GB_mode
# # 23/5/22    .61.211  resnet44  40% 
# 88.39% CUDA_VISIBLE_DEVICES=0 python train.py --model resnet56 --BATCH_SIZE 1024 --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.8 --reclust_pur 0.65 --noisy_pro 0.4 --retention_ratio 1  --GB_mode
# # 23/5/22    .61.211  resnet44  50% 
# 85.76% CUDA_VISIBLE_DEVICES=1 python train.py --model resnet56 --BATCH_SIZE 1024 --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.7 --reclust_pur 0.6 --noisy_pro 0.5 --retention_ratio 1  --GB_mode


# # ##resnet110
# # 23/5/23    .127.75  resnet44 0%
# 94.65% CUDA_VISIBLE_DEVICES=0 python train.py --model resnet110 --BATCH_SIZE 512 --epochs 600 --LR 0.1 --dataset cifar10 --purity 1 --reclust_pur 1 --noisy_pro 0 --retention_ratio 1  --GB_mode
# # # 23/5/23    .127.75  resnet44  10%
# 92.4% CUDA_VISIBLE_DEVICES=0 python train.py --model resnet110 --BATCH_SIZE 1024 --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.95 --reclust_pur 0.8 --noisy_pro 0.1 --retention_ratio 1  --GB_mode
# # # 23/5/23    .127.75  resnet44  20%  
# 90.97% CUDA_VISIBLE_DEVICES=1 python train.py --model resnet110 --BATCH_SIZE 1024 --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.9 --reclust_pur 0.75 --noisy_pro 0.2 --retention_ratio 1  --GB_mode
# # # 23/5/23    .42.178  resnet44  30% 
# 90.06% CUDA_VISIBLE_DEVICES=0 python train.py --model resnet110 --BATCH_SIZE 1024 --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.85 --reclust_pur 0.7 --noisy_pro 0.3 --retention_ratio 1  --GB_mode
# # # 23/5/23    .42.178  resnet44  40% 
# 88.7% CUDA_VISIBLE_DEVICES=1 python train.py --model resnet110 --BATCH_SIZE 1024 --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.8 --reclust_pur 0.65 --noisy_pro 0.4 --retention_ratio 1  --GB_mode
# # # 23/5/22    .61.211  resnet44  50% 
# 84.88% CUDA_VISIBLE_DEVICES=0 python train.py --model resnet110 --BATCH_SIZE 1024 --epochs 1350 --LR 0.1 --dataset cifar10 --purity 0.75 --reclust_pur 0.6 --noisy_pro 0.5 --retention_ratio 1  --GB_mode

# ``
# ############# no GB

# #resnet20 0% 10% 20% 30% 40% 50%   .99  23/5/10     .75 23/5/10
# 92.64%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet20 --BATCH_SIZE 128  --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0 
# 90.49%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet20 --BATCH_SIZE 128  --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.1   
# 88.79%** CUDA_VISIBLE_DEVICES=0 python train.py --model resnet20 --BATCH_SIZE 128  --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.2   

# 87.53%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet20 --BATCH_SIZE 128  --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.3   
# 85.42%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet20 --BATCH_SIZE 128  --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.4   
# 83.58%** CUDA_VISIBLE_DEVICES=1 python train.py --model resnet20 --BATCH_SIZE 128  --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.5   

# ##resnet56
# # 23/5/25    .127.75  resnet44 0%
# CUDA_VISIBLE_DEVICES=0 python train.py --model resnet56 --BATCH_SIZE 128 --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0   
# # 23/5/25    .127.75  resnet44  10%
# CUDA_VISIBLE_DEVICES=1 python train.py --model resnet56 --BATCH_SIZE 128 --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.1
# # 23/5/25    .42.178  resnet44  20%  
# CUDA_VISIBLE_DEVICES=0 python train.py --model resnet56 --BATCH_SIZE 128 --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.2
# # 23/5/25    .42.178  resnet44  30% 
# CUDA_VISIBLE_DEVICES=1 python train.py --model resnet56 --BATCH_SIZE 128 --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.3
# # 23/5/25    .61.211  resnet44  40% 
# CUDA_VISIBLE_DEVICES=0 python train.py --model resnet56 --BATCH_SIZE 128 --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.4
# # 23/5/25    .61.211  resnet44  50% 
# CUDA_VISIBLE_DEVICES=1 python train.py --model resnet56 --BATCH_SIZE 128 --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.5


# ##resnet110
# # # 23/5/25    .127.75  resnet110 0%
# # CUDA_VISIBLE_DEVICES=0 python train.py --model resnet110 --BATCH_SIZE 128 --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0   
# # # 23/5/25    .127.75  resnet110  10%
# # CUDA_VISIBLE_DEVICES=1 python train.py --model resnet110 --BATCH_SIZE 128 --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.1
# # # 23/5/25    .42.178  resnet110  20%  
# # CUDA_VISIBLE_DEVICES=0 python train.py --model resnet110 --BATCH_SIZE 128 --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.2
# # # 23/5/25    .42.178  resnet110  30% 
# # CUDA_VISIBLE_DEVICES=1 python train.py --model resnet110 --BATCH_SIZE 128 --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.3
# # # 23/5/25    .61.211  resnet110  40% 
# # CUDA_VISIBLE_DEVICES=0 python train.py --model resnet110 --BATCH_SIZE 128 --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.4
# # # 23/5/25    .61.211  resnet110  50% 
# # CUDA_VISIBLE_DEVICES=1 python train.py --model resnet110 --BATCH_SIZE 128 --epochs 300 --LR 0.1 --dataset cifar10 --noisy_pro 0.5



# #densenet121 0% 10% 20% 30% 40% 50%   
# # .70.164  23/5/11  
# 95.36%** CUDA_VISIBLE_DEVICES=0 python train.py --model densenet121 --BATCH_SIZE 512  --epochs 600 --LR 0.1 --dataset cifar10 --noisy_pro 0 
# 90.19%** CUDA_VISIBLE_DEVICES=1 python train.py --model densenet121 --BATCH_SIZE 512  --epochs 600 --LR 0.1 --dataset cifar10 --noisy_pro 0.1
# # .42.178 23/5/11
# 85.67%** CUDA_VISIBLE_DEVICES=0 python train.py --model densenet121 --BATCH_SIZE 512  --epochs 600 --LR 0.1 --dataset cifar10 --noisy_pro 0.2
# 82.37%** CUDA_VISIBLE_DEVICES=1 python train.py --model densenet121 --BATCH_SIZE 512  --epochs 600 --LR 0.1 --dataset cifar10 --noisy_pro 0.3   
# # .42.178 23/5/12
# 80.71%** CUDA_VISIBLE_DEVICES=0 python train.py --model densenet121 --BATCH_SIZE 512  --epochs 600 --LR 0.1 --dataset cifar10 --noisy_pro 0.4   
# 74.92%** CUDA_VISIBLE_DEVICES=1 python train.py --model densenet121 --BATCH_SIZE 512  --epochs 600 --LR 0.1 --dataset cifar10 --noisy_pro 0.5   

