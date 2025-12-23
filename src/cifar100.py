'''Train CIFAR100 with PyTorch.'''
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import json
import wandb
from training import train, test
from models import resnet18
from utils import get_lr_scheduler, get_bs_scheduler, get_config_value
from optim.shb import SHB
from optim.nshb import NSHB


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('config_path', type=str, help='path of config file(.json)')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    lr = get_config_value(config, "init_lr")
    epochs = get_config_value(config, "epochs")

    # Dataset Preparation
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(15),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Device Setting
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=100).to(device)

    criterion = nn.CrossEntropyLoss()

    bs_scheduler, total_steps = get_bs_scheduler(config, trainset_length=len(trainset))
    print(f"total_steps: {total_steps}")

    optim_name = get_config_value(config, "optimizer")
    if optim_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optim_name == "shb":
        optimizer = SHB(model.parameters(), lr=lr, momentum=0.9)
    elif optim_name == "nshb":
        optimizer = NSHB(model.parameters(), lr=lr, momentum=0.9)    
    elif optim_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99)
    elif optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    lr_scheduler, lr_step_type = get_lr_scheduler(optimizer, config, total_steps)
    print(optimizer)

    start_epoch = 0
    steps = 0
    batch_size = bs_scheduler.get_batch_size()

    use_wandb = get_config_value(config, "use_wandb")
    if use_wandb:
        wandb_exp_name = get_config_value(config, "optimizer")
        wandb_project_name = "CIFAR-100"
        wandb.init(config=config, project=wandb_project_name, name=wandb_exp_name, entity="XXXXXX")

    for epoch in range(start_epoch, epochs):
        batch_size = bs_scheduler.get_batch_size()
        print(f'batch size: {batch_size}')
        print(f'learning rate: {lr_scheduler.get_last_lr()[0]}')

        train_loss, train_acc, norm_result, last_lr = train(model, device, trainset, optimizer, lr_scheduler, lr_step_type, criterion, batch_size)
        test_loss, test_acc = test(model, device, testloader, criterion)
    
        if lr_step_type == "epoch":
            lr_scheduler.step()
        bs_scheduler.step()

        print(f'Epoch: {epoch + 1}, Steps: {steps}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.2f}%')

        if use_wandb:
            wandb.log({
                    "train_loss": train_loss, "train_acc": train_acc,
                    "norm_result": norm_result,
                    "test_loss": test_loss, "test_acc": test_acc
                    })
