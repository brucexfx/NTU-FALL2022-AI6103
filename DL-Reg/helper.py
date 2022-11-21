# import all libraries

import torch

import os


# Training
def train(epoch, net, criterion, trainloader, scheduler,optimizer,log_step,device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx+1) % log_step == 0:
          print("iteration : %3d, loss : %0.4f, accuracy : %2.2f" % (batch_idx+1, train_loss/(batch_idx+1), 100.*correct/total))
    if scheduler:
        scheduler.step()
    return train_loss/(batch_idx+1), 100.*correct/total

def train_w_DL_Reg(epoch, net, criterion,MSE_criterion, trainloader, scheduler,optimizer,log_step,device,gamma_factor = 1e-12):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        inputs = torch.flatten(inputs, start_dim=1).float()
        if list(inputs.size())[0] >= list(inputs.size())[1]:
            Z = torch.inverse(torch.transpose(inputs, 0, 1) @ inputs + torch.eye(list(inputs.size())[1]).to(device)) @ torch.transpose(inputs, 0,
                                                                                                                1) @ outputs
        else:
            Z = torch.transpose(inputs, 0, 1) @ torch.inverse(inputs @ torch.transpose(inputs, 0, 1)) @ outputs
 
        estim = inputs @ Z
        reg_error = MSE_criterion(estim, outputs) 
        loss = criterion(outputs, targets).float()+gamma_factor*reg_error
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx+1) % log_step == 0:
          print("iteration : %3d, loss : %0.4f, accuracy : %2.2f" % (batch_idx+1, train_loss/(batch_idx+1), 100.*correct/total))
          print("reg term is:",gamma_factor*reg_error.item())
    if scheduler:
        scheduler.step()
    return train_loss/(batch_idx+1), 100.*correct/total


def test(epoch, net, criterion, testloader,device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return test_loss/(batch_idx+1), 100.*correct/total

def save_checkpoint(net, acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth')



