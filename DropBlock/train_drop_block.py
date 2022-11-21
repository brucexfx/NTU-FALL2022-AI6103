# import all libraries
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from helper import *

import random
import matplotlib.pyplot as plt
import numpy as np
import os 

# REPRODUCIBILITY
random.seed(6103)
np.random.seed(6103)
torch.manual_seed(6103)
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

# GPU Test
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
device = torch.device(DEVICE)


import torch
import torch.nn.functional as F
from torch import nn

block_size = 5
drop_prob = 0.1

class DropBlock2D(nn.Module):
    """
    DropBlock: A regularization method for convolutional networks,
    https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            mask = mask.to(x.device)
            block_mask = self._compute_block_mask(mask)
            out = x * block_mask[:, None, :, :]
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)

class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=int(nr_steps))

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1

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
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # This is the "stem"
        # For CIFAR (32x32 images), it does not perform downsampling
        # It should downsample for ImageNet
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # four stages with three downsampling
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0.,
            stop_value=drop_prob,
            nr_steps=5e3
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        self.dropblock.step() #  increment number of iterations
        
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.dropblock(self.layer1(out))
        out = self.dropblock(self.layer2(out))
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def main():

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)


    trainset,valset = torch.utils.data.random_split(trainset, [40000,10000], generator=torch.Generator().manual_seed(6103))

    # make deep copy and remove augmentations
    valset_remove_aug = copy.deepcopy(valset)
    valset_remove_aug.dataset.transforms = transform_test

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=256, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(
        valset_remove_aug, batch_size=256, shuffle=True, num_workers=4)

    # testset = torchvision.datasets.CIFAR10(
    #     root='./data', train=False, download=True, transform=transform_test)

    # # we can use a larger batch size during test, because we do not save 
    # # intermediate variables for gradient computation, which leaves more memory
    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=256, shuffle=False, num_workers=2)

    fig_loss,ax_loss = plt.subplots(1,1,figsize=(10,5))
    fig_acc,ax_acc = plt.subplots(1,1,figsize=(10,5))
    fig_lr,ax_lr = plt.subplots(1,1,figsize=(10,5))
    
    config = {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 10e-4,
        'epoch': 300
    }

    metric = []

    print(f"lr: {config['lr']} momentum: {config['momentum']} weight_decay: {config['weight_decay']} Epoch: {config['epoch']}")
    net = ResNet50().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr = config['lr'],
                            momentum=config['momentum'], weight_decay=config['weight_decay'])
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,75)
    # scheduler = None
    train_losses, train_accs,test_losses, test_accs = [], [], [],[]
    lr_ls = []

    for epoch in range(1, config['epoch']+1):   
        train_loss, train_acc = train(epoch, net, criterion, trainloader, scheduler,optimizer,log_step=50,device=device)
        test_loss, test_acc = test(epoch, net, criterion, valloader,device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        print(("Epoch : %3d, training loss : %0.4f, training accuracy : %2.2f, test loss " + \
            ": %0.4f, test accuracy : %2.2f") % (epoch, train_loss, train_acc, test_loss, test_acc))
        print(f'LR : {optimizer.param_groups[0]["lr"]}')
        lr_ls.append(optimizer.param_groups[0]["lr"])
    metric.append([train_losses,test_losses,train_accs,test_accs])
    ax_loss.plot(range(1,config['epoch']+1),train_losses,label = f"Train")
    ax_loss.plot(range(1,config['epoch']+1),test_losses,"--",label = f"Val")

    ax_acc.plot(range(1,config['epoch']+1),train_accs,label = f"Train")
    ax_acc.plot(range(1,config['epoch']+1),test_accs,"--",label = f"Val")

    ax_lr.plot(range(1,config['epoch']+1),lr_ls,"--",label = f"LR")
    # del net
    # del criterion
    # del optimizer
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()

    ax_lr.set_xlabel('Iteration')
    ax_lr.set_ylabel('LR')    
    ax_lr.set_title("LR vs vs Epochs")
    ax_lr.legend()

    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')    
    ax_loss.set_title("Loss vs vs Epochs")
    ax_loss.legend()

    ax_acc.set_xlabel('Iteration')
    ax_acc.set_ylabel('Accuracy') 
    ax_acc.set_title("Accuracy vs Epochs")
    ax_acc.legend()
    
    fig_loss.savefig("loss_10_01_bs5.jpg",dpi=600)
    fig_acc.savefig("acc_10_01_bs5.jpg",dpi=600)
    fig_lr.savefig("lr_10_01_bs5.jpg",dpi=600)
    
    torch.save(metric,"metric_10_drop_0.1_bs5.pth")
    torch.save(net.state_dict(),"model_10_drop_0.1_bs5.pth")
    
if __name__ == "__main__":
    main()