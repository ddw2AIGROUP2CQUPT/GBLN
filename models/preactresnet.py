import torch
import torch.nn as nn
import torch.nn.functional as F
# from lib.normalize import Normalize
from myrelu import GBNR
from torch.autograd import Variable
cpu_device = torch.device("cpu")
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2((F.relu(self.bn2(out))))
        out += shortcut
        return out

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, nb_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_64 = nn.Linear(512*block.expansion, 64)
        self.fc_10 = nn.Linear(64, nb_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, target, index, original_target, GB_layer=False, purity=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc_64(out)

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

        out = self.fc_10(out)
        # end_time = time.time()
        # print("总时：",end_time-since_time)
        return out, target, relabel
        

def PreActResNet18(low_dim=128):
    return ResNet(PreActBlock, [2,2,2,2], low_dim)

def PreActResNet34(low_dim=128):
    return ResNet(PreActBlock, [3,4,6,3], low_dim)

def PreActResNet50(low_dim=128):
    return ResNet(PreActBottleneck, [3,4,6,3], low_dim)

def PreActResNet101(low_dim=128):
    return ResNet(PreActBottleneck, [3,4,23,3], low_dim)

def PreActResNet152(low_dim=128):
    return ResNet(PreActBottleneck, [3,8,36,3], low_dim)


# def test():
#     net = PreActResNet18()
#     y = net(Variable(torch.randn(10,3,32,32)))
    print(y.size())

# test()
