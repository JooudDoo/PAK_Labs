# %%
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import train_test_split

from torchvision import datasets, transforms
import torchinfo 

# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip()]
)
batch_size = 128
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
epochs = 35

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_set = datasets.CIFAR10(
    root='./dataset', train=True, download=True, transform=train_transform)
valid_set = datasets.CIFAR10(
    root='./dataset', train=True, download=True, transform=train_transform)
test_set = datasets.CIFAR10(
    root='./dataset', train=False, download=True, transform=transform)

train_idx, valid_idx   = train_test_split(range(len(train_set)), test_size=0.2, train_size=0.8)
# idxs = list(range(len(train_set)))
# split = int(np.floor(0.2*len(train_set)))
# valid_idx, train_idx = idxs[:split], idxs[split:]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
num_workers = 4

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(
    valid_set, batch_size=batch_size, num_workers=num_workers, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, num_workers=num_workers)


# %%
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.shuffle1 = nn.Sequential(nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        ), nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.shuffle2 = nn.Sequential(nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        ), nn.Sequential(
            nn.Conv2d(512, 512,3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True))
        )

        self.res = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, len(classes))
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.shuffle1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.shuffle2(x) + x
        x = self.res(x)
        return x


# %%
model = Model()

# try:
print(torchinfo.summary(model, input_size=(batch_size, 3, 32, 32)))
# except RuntimeError as a:
#     print(a)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)

model.to(device)


criterion = nn.CrossEntropyLoss()
from torch.optim.lr_scheduler import ReduceLROnPlateau
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

scheduler = ReduceLROnPlateau(optim, 'min')


# %%
def train_step():
    model.train()
    running_loss = 0.
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optim.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optim.step()
        running_loss += loss
    with torch.no_grad():
        train_loss = running_loss / len(train_loader)
    return train_loss.item()


def valid_step():
    model.eval()
    correct_total = 0
    running_loss = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            prediction = output.max(dim=1, keepdim=True)[1]
            correct_total += prediction.eq(labels.view_as(prediction)).sum()
            loss = criterion(output, labels)
            running_loss += loss
        valid_loss = running_loss / len(valid_loader)
        accuracy = correct_total / len(valid_sampler)
        return valid_loss.item(), accuracy.item()


train_losses = []
valid_losses = []
valid_accs = []


# %%
from tqdm import tqdm

best_loss = 10000

for epoch in (pbar := tqdm(range(epochs))):
    train_loss = train_step()
    valid_loss, valid_acc = valid_step()
    scheduler.step(valid_loss)

    if valid_loss < best_loss and  epoch > 3:
        best_loss = valid_loss
        print(f"Saved loss with acc: {valid_acc} | loss: {valid_loss}")
        torch.save(model.state_dict(), f"./cifar_temp{epoch}.pth")

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    valid_accs.append(valid_acc)

    pbar.set_description(
        f'Acc: {valid_acc:.4f}  Avg. train/valid loss: {train_loss:.4f}/{valid_loss:.4f}')
    

PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)


# %%
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 4))

plt.plot(train_losses, label='train')
plt.plot(valid_losses, label='valid')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Loss')


# %%
fig = plt.figure(figsize=(6, 4))

plt.plot(valid_accs)
plt.yticks(np.arange(0.35, 1, 0.05))
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Accuracy')


# %%
def test_accuracy_per_class(net, testloader):
    correct_pred = {classname: 0 for classname in train_set.classes}
    total_pred = {classname: 0 for classname in train_set.classes}

    with torch.no_grad():
        net.eval()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            predicted = torch.max(outputs.data, 1)[1]

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[train_set.classes[label]] += 1
                total_pred[train_set.classes[label]] += 1
    
    accuracy_per_class = {classname: 0 for classname in train_set.classes}
    for classname, correct_count in correct_pred.items():
        accuracy = (100 * float(correct_count)) / total_pred[classname]
        accuracy_per_class[classname] = accuracy

    return accuracy_per_class

accuracy_per_class = test_accuracy_per_class(model, test_loader)
for classname, accuracy in accuracy_per_class.items():
    print(f'{classname:12s} {accuracy:.2f} %')





