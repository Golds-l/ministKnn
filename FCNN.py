import time

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import wandb

from mnistRead import calAccuracy

EPOCH = 200
LR = 0.004
BATCH_SIZE = 512


class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.FC1 = nn.Sequential(
            nn.Linear(28 * 28 * 1, 512),
            nn.ReLU()
        )
        self.FC2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.FC3 = nn.Sequential(
            nn.Linear(128, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        return x


def train(net, optimizer, dataloader):
    net.train()
    losses = []
    for idx, (input, label) in enumerate(dataloader):
        input, label = input.cuda(), label.cuda()
        optimizer.zero_grad()
        output = net(input)
        loss = F.cross_entropy(output, label)
        losses.append(round(loss.item(), 5))
        loss.backward()
        optimizer.step()
    return sum(losses) / len(losses)


def validation(net, dataloader):
    net.eval()
    losses = []
    accNum = 0
    with torch.no_grad():
        for idx, (input, label) in enumerate(dataloader):
            input, label = input.cuda(), label.cuda()
            output = net(input)
            loss = F.cross_entropy(output, label)
            accNum += calAccuracy(label, output)
            losses.append(round(loss.item(), 5))
    return sum(losses) / len(losses), accNum


if __name__ == "__main__":
    wandb.init(project="FCNN", config={"epochs": EPOCH, "batch_size": BATCH_SIZE, "lr": LR})
    modelLoss = float("inf")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    datasetMNIST = torchvision.datasets.MNIST(root="./mnistData", train=True, download=True, transform=transforms)
    dataloaderMNIST = DataLoader(datasetMNIST, shuffle=True, batch_size=BATCH_SIZE)
    valDataset = torchvision.datasets.MNIST(root="./mnistData", train=False, download=True, transform=transforms)
    valDataloader = DataLoader(valDataset, shuffle=True, batch_size=BATCH_SIZE)
    network = FCNN().to(device)
    opt = optim.SGD(network.parameters(), lr=LR)
    for i in range(EPOCH):
        wandb.watch(network)
        tE = time.time()
        trainLoss = train(network, opt, dataloaderMNIST)
        wandb.log({"train loss": trainLoss})
        if i % 5 == 0:
            valLoss, accCount = validation(network, valDataloader)
            wandb.log({"val loss": valLoss})
            if valLoss < modelLoss:
                modelLoss = valLoss
                torch.save(network.state_dict(), "./models/FCNN/best.pt")
                print(f"epoch:{i}  trainLoss:{trainLoss}  valLoss:{valLoss} acc:{accCount} time:{time.time() - tE}s SAVED!")
                continue
            print(f"epoch:{i}  trainLoss:{trainLoss}  valLoss:{valLoss} acc:{accCount} time:{time.time() - tE}s")
            continue
        print(f"epoch:{i}  trainLoss:{trainLoss}  time:{time.time() - tE}s")
