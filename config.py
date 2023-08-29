import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--BATCH_SIZE', type=int, default=1024, help='resnet20 1024  densenet121 512')                                   # 一次训练所取的样本数  128 -> 512
parser.add_argument('--LR', type=float, default=0.1)                                         # 学习率  0.1 -> 0.01
parser.add_argument('--weight_decay', type=float, default=5e-4)                              # 权重衰减，惩罚系数
parser.add_argument('--momentum', type=float, default=0.9)                                   # 动量
parser.add_argument('--epochs', type=int, default=1350)                                       # 训练次数：res:200; dense:300
parser.add_argument('--print_intervals', type=int, default=20)                              # 测试间隔：1024-20,10000-2

# parser.add_argument('--evaluation', type=bool, default=False)
parser.add_argument('--checkpoints', type=str, default=None, help='model checkpoints path')  # 模型参数保存
# parser.add_argument('--device_num', type=int, default=1)

parser.add_argument('--dataset', type=str, default='cifar10', help="数据集名")
parser.add_argument('--model', type=str, default='preactresnet18', help="模型[resnet20,densenet121]")
parser.add_argument('--data_path', type=str, default="/home/ubuntu/workplace/zh/dataset/cifar-10-batches-py", help="数据集地址")
parser.add_argument('--purity', type=float, default=1)
parser.add_argument('--reclust_pur', type=float, default=1, help='内部再聚类纯度')
parser.add_argument('--noisy_pro', type=float, default=0, help="噪声比例")
parser.add_argument('--retention_ratio', type=float, default=1, help="保留比例")
parser.add_argument('--noisy_mode', type=str, default='sym', help="sym——对称噪声  asym——非对称噪声")
parser.add_argument('--GB_mode', default='False', action='store_true', help="粒球层")
parser.add_argument('--num', default=0, type=float, help='第几次训练')

args = parser.parse_args()

# 5.8:'/root/Dataset/cifar10/'
# 5.7:"/home/ubuntu/workplace/zg/data/cifar10/"
# 5.81:'/root/zzg/cifar10-resnet/cifar10/'
# benji: 'E:/DATAset/cifar10/'
