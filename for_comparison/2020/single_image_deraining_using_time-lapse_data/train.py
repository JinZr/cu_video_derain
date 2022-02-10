import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train():
    # loop over the dataset multiple times
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Loss: {}'.format(running_loss)

    print('Finished Training')
