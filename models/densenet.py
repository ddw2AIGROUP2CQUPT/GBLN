# from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from myrelu import GBNR
import os
import random
import numpy as np
import time
from collections import OrderedDict
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
cpu_device = torch.device("cpu")
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _DenseLayer(nn.Sequential):
    # num_input_features作为输入特征层的通道数， growth_rate增长率， bn_size输出的倍数一般都是4， drop_rate判断是都进行dropout层进行处理
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()

        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1',
                        nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))

        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2',
                        nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


# 定义Denseblock模块
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module("denselayer %d" % (i + 1), layer)


# 定义Transition层
# 负责将Denseblock连接起来，一般都有0.5的维道（通道数）的压缩
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(2, stride=2))


# 实现DenseNet网络
class DenseNet(nn.Module):

    def _apply_to_exp_pool(self, sample_x, sample_label):
        n = len(self.ball_pool)
        # 超出经验池上限处理
        if n >= self.max_memory:
            pop_num = (n - self.max_memory) + len(sample_x)
            # 两种策略丢弃历史经验，先入先出，随机丢弃
            if self.forget_stratege == 'FIFO':
                self.ball_pool = self.ball_pool[pop_num:]
                self.target_pool = self.target_pool[pop_num:]
            elif self.forget_stratege == 'Random':
                idx = np.arange(0, n)
                keep_idx = np.random.choice(idx, size=n - pop_num)
                self.ball_pool = self.ball_pool[keep_idx]
                self.target_pool = self.target_pool[keep_idx]
        self.ball_pool.extend(sample_x)  # = np.concatenate([self.ball_pool, balls], axis=0)
        self.target_pool.extend(sample_label)  # = np.concatenate([self.target_pool, target], axis=0)

        # 从经验池随机抽取

    def _get_from_exp_pool(self, ):
        n = len(self.ball_pool)
        if n < self.memory:
            # 少于期望数量的经验则全部抽取
            return torch.Tensor(self.ball_pool), torch.Tensor(self.target_pool)
        else:
            # 随机抽取
            # idx = np.arange(0, n)
            # choice_idx = np.random.choice(idx, self.memory)
            # return torch.Tensor(self.ball_pool(choice_idx)), torch.Tensor(self.target_pool(choice_idx))
            together_ = [[a, b] for a, b in zip(self.ball_pool, self.target_pool)]
            selected = random.sample(together_, self.memory)
            ball_ = []
            target_ = []
            for x in selected:
                ball_.append(x[0])
                target_.append(x[1])
            return torch.Tensor(ball_), torch.Tensor(target_)
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 26), num_init_features=64, bn_size=4,
                 comparession_rate=0.5, drop_rate=0, num_classes=10):
        super(DenseNet, self).__init__()
        # 前面 卷积层+最大池化
        self.fc_64 = nn.Linear(64, num_classes)
        self.fc_1000 = nn.Linear(1000, 64)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(3, stride=1, padding=1))

        ]))
        # Denseblock
        # print("fu1111")
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)

            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features += num_layers * growth_rate  # 确定一个DenseBlock输出的通道数

            if i != len(block_config) - 1:  # 判断是不是最后一个Denseblock
                transition = _Transition(num_features, int(num_features * comparession_rate))
                self.features.add_module("transition%d" % (i + 1), transition)
                num_features = int(num_features * comparession_rate)  # 为下一个DenseBlock的输出做准备

        # Final bn+ReLu
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        # classification layer
        self.classifier = nn.Linear(num_features, 1000)

        # params initalization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, target, index, original_target, GB_layer=False, purity=None):
        features = self.features(x)
        out = F.avg_pool2d(features, 3, stride=2).view(features.size(0), -1)
        out = self.classifier(out)
        out = out.reshape(out.size(0), -1)  # b,64
        out = self.fc_1000(out)

        if GB_layer == True:
            out = torch.cat((target.reshape(-1, 1), out), dim=1)  # [batchsize, 1(label)+64]
            out = torch.cat((index.reshape(-1, 1), out), dim=1)  # [b, index+label+64]
            out = torch.cat((original_target.reshape(-1, 1), out), dim=1)  # [b, ori+index+label+64]
            pur_tensor = torch.Tensor([[purity]] * out.size(0))
            out = torch.cat((pur_tensor.to(device), out), dim=1)  # [:,pur+ori+index+label+64]
            out, target = GBNR.apply(out.to(cpu_device))  # 聚球
            out, target = out.to(device), target.to(device)
            # print("sample_index:", sample_index)
            # 按照球个数，将样本划分为每个球

        out = self.fc_64(out)

        return out, target, index


def densenet121(pretrained=False, **kwargs):
    """DenseNet121"""
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    return model


if __name__ == '__main__':
    # f1：structure and params of each layer (include two conv in shortcut)
    model = densenet121()