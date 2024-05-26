import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

# class ResnetClassifier(nn.Module):
#     def __init__(self,num_classes=1000, use_pretrained=True):
#         self.backbone = models.resnet18(pretrained=use_pretrained)
#         in_features = self.backbone.fc.in_features
#         self.fc = nn.Linear(in_features, num_classes) 

#     def forward(self, x):
#         out = self.backbone(x)
#         out = self.fc(out)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        depth =64#64
        self.in_planes = depth

        self.conv1 = nn.Conv2d(3, depth, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(depth)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, depth, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, depth*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, depth*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, depth*8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        # self.linear = nn.Linear(256*block.expansion, num_classes)
        in_features = int(512*block.expansion)#*tensor_size[1]/32)
        # print(in_features)
        # in_features=4608#115200 #57600#
        # in_features = 18432#131072
        self.linear = nn.Linear(in_features, num_classes)#5376

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.relu(out)
        out = self.maxpool(out)
        # print('103', out.size())
        out = self.layer1(out)
        # print('105', out.size())
        out = self.layer2(out)
        # print('107', out.size())
        out = self.layer3(out)
        # print('109', out.size())
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        # print('111', out.size())
        # out = F.avg_pool2d(out, 1)
        # print('113', out.size())
        # out = out.view(out.size(0), -1)
        # print('115', out.size())
        out = self.dropout(out)
        # print('117', out.size())
        out = self.linear(out)
        # print('119', out.size())
        return out

def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2],num_classes=num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3],num_classes=num_classes)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3],num_classes=num_classes)


def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3],num_classes=num_classes)


def ResNet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3],num_classes=num_classes)