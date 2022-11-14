# import all libraries
import copy
import torch
import torch.nn as nn
import torch.optim as optim


import torchvision
import torchvision.transforms as transforms

from helper import *
from resnet import *

import random
import matplotlib.pyplot as plt
import numpy as np
import os 

# REPRODUCIBILITY
random.seed(6103)
np.random.seed(6103)
torch.manual_seed(6103)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
# os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2" 报错换这个


# GPU Test
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
device = torch.device(DEVICE)

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

    for epoch in range(1, config['epoch']+1):   
        train_loss, train_acc = train(epoch, net, criterion, trainloader, scheduler,optimizer,log_step=50,device=device)
        test_loss, test_acc = test(epoch, net, criterion, valloader,device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(("Epoch : %3d, training loss : %0.4f, training accuracy : %2.2f, test loss " + \
            ": %0.4f, test accuracy : %2.2f") % (epoch, train_loss, train_acc, test_loss, test_acc))
    metric.append([train_losses,test_losses,train_accs,test_accs])
    ax_loss.plot(range(1,config['epoch']+1),train_losses,label = f"Train")
    ax_loss.plot(range(1,config['epoch']+1),test_losses,"--",label = f"Val")

    ax_acc.plot(range(1,config['epoch']+1),train_accs,label = f"Train")
    ax_acc.plot(range(1,config['epoch']+1),test_accs,"--",label = f"Val")
    # del net
    # del criterion
    # del optimizer
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()

    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')    
    ax_loss.set_title("Loss vs vs Epochs")
    ax_loss.legend()

    ax_acc.set_xlabel('Iteration')
    ax_acc.set_ylabel('Accuracy') 
    ax_acc.set_title("Accuracy vs Epochs")
    ax_acc.legend()
    fig_loss.savefig("loss.jpg",dpi=600)
    fig_acc.savefig("acc.jpg",dpi=600)
    torch.save(metric,"metric.pth")
    
if __name__ == "__main__":
    main()