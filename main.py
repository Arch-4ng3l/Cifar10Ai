import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from PIL import Image, ImageOps
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
import random
classes = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2, 
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8, 
    "truck": 9
}


def one_hot(N):
    arr = np.zeros(10)
    arr[N] = 1
    return arr

dir = os.listdir("./train/")
train_dir = os.listdir("./test/")
label_dir = pd.read_csv("trainLabels.csv")['label']
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.49, 0.48, 0.44), (0.2, 0.2, 0.2))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49, 0.48, 0.44), (0.2, 0.2, 0.2))
])
Batched_Data = []
Test_Data = []

def load_imgs(Batched_Data, dir, label_dir, batch_size=32):
    Labels = []
    Data = []
    for file in tqdm(dir): 
        img = Image.open("./train/" + file)
        num = int(file.replace(".png", ""))
        label = label_dir[num-1]
        if batch_size == 1:
            img = transform_test(img)
        else:
            img = transform(img)

        label = classes[label]
        Data.append(img)
        Labels.append(one_hot(label))
        if len(Data) == batch_size:
            Batched_Data.append((torch.stack(Data), torch.Tensor(np.array(Labels))))
            Data = []
            Labels = []

load_imgs(Batched_Data, dir[:len(dir) - 5000], label_dir)
load_imgs(Test_Data, dir[len(dir)-5000:], label_dir, batch_size=1)






class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, X):
        res = X
        out = self.conv1(X)
        out = self.conv2(out)
        if self.downsample:
            res = self.downsample(X)

        out = out + res
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 10)
        self.dropout = nn.Dropout()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
                nn.Conv2d(planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, X):
        X = self.conv1(X)
        X = self.maxpool(X)
        X = self.layer0(X)
        X = self.dropout(X)
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.dropout(X)
        X = self.layer3(X)

        X = self.flatten(X)
        X = self.fc(X)
        return X



model = ResNet(ResBlock, [3, 3, 6, 3])


optim = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
criterion = nn.CrossEntropyLoss()


length = len(Batched_Data)
losses = []
plt.ion()
plt.show()
ax = plt.gca()
ax.legend(["Training", "Test"])
ax.set_ylim((0, 3))


train_losses = []
losses2 = []
test_losses = []

def add_train_loss(loss):
    train_losses.append(loss.detach().numpy())

def add_test_loss(loss):
    test_losses.append(loss.detach().numpy())




def train(epoch):
    print(f"Epoch: {epoch}")
    model.train()
    comb_loss = 0
    random.shuffle(Batched_Data)
    for imgs, labels in tqdm(Batched_Data):
        optim.zero_grad()

        imgs = Variable(imgs)
        labels = Variable(labels)

        out = model(imgs)
        loss = criterion(out, labels)

        comb_loss += loss

        loss.backward()
        optim.step()

    add_train_loss(comb_loss/length)
    print(f"Loss: {comb_loss/length}")


def test():
    model.eval()
    corrects, wrong = 0, 0
    comb_loss = 0
    for i, (imgs, labels) in enumerate(Test_Data):

        imgs = Variable(imgs)
        label = np.argmax(labels)

        labels = Variable(labels)
        out = model(imgs)
        loss = criterion(out, labels)
        comb_loss += loss
        guess = torch.argmax(out)
        if guess == label:
            corrects += 1
        else:
            wrong += 1
    add_test_loss(comb_loss/len(Test_Data))
    print(f"Accuracy = {corrects/(corrects + wrong) * 100}%")


    

for epoch in range(1, 30):
    train(epoch)
    with torch.no_grad():
        test()
    plt.plot(train_losses, color="red")
    plt.plot(test_losses, color="blue")
    plt.pause(0.0001)
    torch.save(model.state_dict(), "model.pt")
