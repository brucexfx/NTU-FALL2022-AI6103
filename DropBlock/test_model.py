from helper import *
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from train_drop_block import ResNet50

def get_test_loader(data_dir, batch_size, shuffle=False):
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=2)

    return test_loader

if __name__ == "__main__":
    # main body
    DEVICE = "cuda"
    device = torch.device(DEVICE)
    test_loader = get_test_loader("./data", batch_size=256, shuffle=False)
    net = ResNet50().to(device)
    pretrain = torch.load("model_10_drop_0.1_bs5.pth")
    net.load_state_dict(pretrain)
    criterion = nn.CrossEntropyLoss().to(device)
    test_loss, test_acc = test(1, net, criterion, test_loader, device)
    print(("test loss " + ": %0.4f, test accuracy : %2.2f") % (test_loss, test_acc))
    
    