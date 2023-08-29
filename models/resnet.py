"""
ResNet Image Classfication for Cifar-10 with PyTorch
Author 'ZgZhuang'
"""
# from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from myrelu import GBNR
import os
import random
import numpy as np
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
cpu_device = torch.device("cpu")
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

cost = nn.CrossEntropyLoss().to(device)
best_out = None
best_target = None


class BasicBlock(nn.Module):
    # 通道放大倍数
    expansion = 1  # BasicBlock中每个残差结构输出维度和输入维度相同

    def __init__(self, in_planes, planes, stride=1):
        # 调用父类初始化函数
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:  # 符合条件时，shortcut采用1x1 conv 来匹配通道数一致
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)  # Residual Structure：relu() after 'out+shortcut'
        return out


class ResNet_CiFar(nn.Module):
    def __init__(self, block, num_blocks, memory=100, max_memory=10000, forget_stratege="FIFO", num_classes=10):
        super(ResNet_CiFar, self).__init__()
        self.in_planes = 16
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self.make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 64, num_blocks[2], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc_64 = nn.Linear(64, num_classes)

    def make_layer(self, block, planes, num_blocks, stride):
        """
        :param block: BasicBlock or Bottleneck
        :param planes: out channel number of this layer
        :param num_blocks: how many blocks per layer
        :param stride: the stride of the first block of this layer(1 or 2), other blocks will always be 1
        :return: a resnet block layer with several blocks
        """
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1,1]  [2,1,1]  [2,1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, target, index, original_target, GB_layer=False, purity=None):
        
        # since_time = time.time()
        out = self.conv0(x)  # b,16,32,32
        out = self.layer1(out)  # b,16,32,32
        out = self.layer2(out)  # b,32,16,16
        out = self.layer3(out)  # b,64,8,8
        out = self.avg_pool(out)  # b,64,1,1
        out = out.reshape(out.size(0), -1)  # b,64
        # encode_time = time.time()
        relabel = None
        if GB_layer == True:
            out = torch.cat((target.reshape(-1, 1), out), dim=1)  # [batchsize, 1(label)+64]
            out = torch.cat((index.reshape(-1, 1), out), dim=1)    # [b, index+label+64]
            out = torch.cat((original_target.reshape(-1, 1), out), dim=1)  # [b, ori+index+label+64]
            pur_tensor = torch.Tensor([[purity]] * out.size(0))
            out = torch.cat((pur_tensor.to(device), out), dim=1)  # [:,pur+ori+index+label+64]
            # preliqiu_time = time.time()
            out, target, relabel= GBNR.apply(out.to(cpu_device))  # 聚球
            # liqiu_time = time.time()
            out, target, relabel= out.to(device), target.to(device), relabel.to(device)
            
            # togpu_time = time.time()
            # print("编码时间：",encode_time-since_time)
            # print("准备粒球时间：",preliqiu_time-encode_time)
            # print("粒球时间：",liqiu_time-preliqiu_time)
            # print("转向gpu时间：",togpu_time-liqiu_time)
            # print("sample_index:", sample_index)
            # 按照球个数，将样本划分为每个球
            # index = []
            # for i in chaifen_id:
            #     index.append(sample_index[:i])
            #     sample_index = sample_index[i:]
            # # print("index:", index)
            # print("本次加入经验池球数：", len(index))

        out = self.fc_64(out)
        # end_time = time.time()
        # print("总时：",end_time-since_time)
        return out, target, relabel


"""
粒球 
输入：1(label)+64维
输出：64+1(半径),1(label)
"""


def resnet20(**kwargs):
    return ResNet_CiFar(BasicBlock, [3, 3, 3], **kwargs)

def resnet32(**kwargs):
    return ResNet_CiFar(BasicBlock, [5, 5, 5], **kwargs)

def resnet44(**kwargs):
    return ResNet_CiFar(BasicBlock, [7, 7, 7], **kwargs)

def resnet56(**kwargs):
    return ResNet_CiFar(BasicBlock, [9, 9, 9], **kwargs)

def resnet110(**kwargs):
    return ResNet_CiFar(BasicBlock, [18, 18, 18], **kwargs)
if __name__ == '__main__':
    # f1：structure and params of each layer (include two conv in shortcut)
    model = resnet20()
    # summary(model, (3, 32, 32))
