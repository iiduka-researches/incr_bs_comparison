import torch
from .get_full_grad_list import get_full_grad_list


def train(model, device, trainset, optimizer, lr_scheduler, lr_step_type, criterion, batch_size):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if lr_step_type == "step":
            lr_scheduler.step()

    train_loss = total_loss / (batch_idx + 1)
    train_acc = 100. * correct / total
    
    p_norm = get_full_grad_list(model, trainset, optimizer, batch_size, device)

    last_lr = lr_scheduler.get_last_lr()[0]

    return train_loss, train_acc, p_norm, last_lr
