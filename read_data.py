import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from cifar import CIFAR10,CIFAR100
from torch.utils.data import DataLoader   # 导入下载通道
import numpy as np
import matplotlib.pyplot as plt
from config import args
# from self_define.mydataloader import *


def read_cifar10(batchsize, data_dir, noisy_pro):

    transform_train = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),        # 填充后裁剪
                                    transforms.RandomHorizontalFlip(p=0.5),      # 水平翻转
                                    transforms.ToTensor(),                       # 转换成tensor的同时,进行了归一化
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

    transform_test = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

    '''
    torchvision.transforms.Compose类看作一种容器，能够同时对多种数据变换进行组合。传入的参数是一个列表，列表中的元素就是对载入的数据进行的
    各种变换操作。在经过标准化变换之后，数据全部符合均值为0、标准差为1的标准正态分布。
    # Q1常见数据增强操作：填充、裁剪、水平/竖直翻转、旋转角度等等，所有操作要在ToTensor前进行
    # Q2数据归一化问题：ToTensor是把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，
                                                                                  取值范围是[0,1.0]的torch.FloatTensor
    # Q3均值标准差问题：mean和std各有三个值,代表RGB3个通道,可以自己求数据集图片在通道的均值方差
                               
    '''
    # 数据加载
    # data_train = CIFAR10(root=data_dir,
    #                               train=True,
    #                               transform=transform_train,
    #                               download=True)
    data_train = CIFAR10(root=data_dir,
                         train=True,
                         transform=transform_train,
                         download=False,
                         noisy_pro=noisy_pro)
    data_test = datasets.CIFAR10(root=data_dir,
                        train=False,
                        transform=transform_test,
                        download=False)

    # 数据装载和数据预览
    data_loader_train = DataLoader(dataset=data_train,
                                   batch_size=batchsize,
                                   shuffle=True,         # 打乱数据
                                   pin_memory=True, )    # K1
                                   # drop_last=True, )   # K2
                                   # num_workers=8)

    data_loader_test = DataLoader(dataset=data_test,
                                  batch_size=1000,
                                  shuffle=False,
                                  pin_memory=True,)
                                  # drop_last=True,)
                                  # num_works=8)

    return data_loader_train, data_loader_test, data_train  # , data_test


def read_cifar100(batchsize, data_dir, noisy_pro):

    transform_train = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),        # 填充后裁剪
                                    transforms.RandomHorizontalFlip(p=0.5),      # 水平翻转
                                    transforms.ToTensor(),                       # 转换成tensor的同时,进行了归一化
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

    transform_test = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

    '''
    torchvision.transforms.Compose类看作一种容器，能够同时对多种数据变换进行组合。传入的参数是一个列表，列表中的元素就是对载入的数据进行的
    各种变换操作。在经过标准化变换之后，数据全部符合均值为0、标准差为1的标准正态分布。
    # Q1常见数据增强操作：填充、裁剪、水平/竖直翻转、旋转角度等等，所有操作要在ToTensor前进行
    # Q2数据归一化问题：ToTensor是把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，
                                                                                  取值范围是[0,1.0]的torch.FloatTensor
    # Q3均值标准差问题：mean和std各有三个值,代表RGB3个通道,可以自己求数据集图片在通道的均值方差
                               
    '''
    # 数据加载
    # data_train = CIFAR10(root=data_dir,
    #                               train=True,
    #                               transform=transform_train,
    #                               download=True)
    data_train = CIFAR100(root=data_dir,
                         train=True,
                         transform=transform_train,
                         download=True,
                         noisy_pro=noisy_pro)
    data_test = datasets.CIFAR100(root=data_dir,
                        train=False,
                        transform=transform_test,
                        download=True
                        )

    # 数据装载和数据预览
    data_loader_train = DataLoader(dataset=data_train,
                                   batch_size=batchsize,
                                   shuffle=True,         # 打乱数据
                                   pin_memory=True, )    # K1
                                   # drop_last=True, )   # K2
                                   # num_workers=8)

    data_loader_test = DataLoader(dataset=data_test,
                                  batch_size=250,
                                  shuffle=False,
                                  pin_memory=True,)
                                  # drop_last=True,)
                                  # num_works=8)

    return data_loader_train, data_loader_test, data_train  # , data_test

if __name__ == '__main__':
    data_loader_train, data_loader_test = read_cifar100(args.BATCH_SIZE, args.data_dir)  # 载入数据
    digit = data_loader_train.dataset.data[2]
    # plt.imshow(digit, cmap=plt.cm.binary)
    # plt.show()
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(classes[data_loader_train.dataset.targets[1]])

# if __name__ == '__main__':
#     data_loader_train, data_loader_test = read_cifar10(args.BATCH_SIZE, args.data_dir)  # 载入数据
#     digit = data_loader_train.dataset.data[2]
#     print(len(data_loader_train))
#     # plt.imshow(digit, cmap=plt.cm.binary)
#     # plt.show()
#     classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#     print(classes[data_loader_train.dataset.targets[1]])



