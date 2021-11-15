import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision

import wandb

EPOCH = 20
LR = 0.0004
BATCH_SIZE = 8


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU()
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1))
        )
        self.FC = nn.Sequential(
            nn.Linear(in_features=120 * 1 * 1, out_features=84 * 1 * 1),
            nn.Linear(in_features=84 * 1 * 1, out_features=10)
        )

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = x.view(-1, 120)
        x = self.FC(x)
        x = F.log_softmax(x, dim=1)
        return x


def train(net, optimizer, dataloader):
    net.train()
    losses = []
    for idx, (input, label) in enumerate(dataloader):
        input, label = input.cuda(), label.cuda()
        optimizer.zero_grad()
        output = net(input)
        loss = F.nll_loss(output, label)
        losses.append(round(loss.item(), 5))
        loss.backward()
        optimizer.step()

    return sum(losses) / len(losses)


if __name__ == "__main__":
    wandb.init(project="LeNet")
    wandb.config = {
        "learning_rate": LR,
        "epochs": EPOCH,
        "batch_size": BATCH_SIZE
    }
    modelLoss = float("inf")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    datasetMNIST = torchvision.datasets.MNIST(root="./ministData", train=True, download=True, transform=transforms)
    dataloaderMNIST = DataLoader(datasetMNIST, shuffle=True, batch_size=BATCH_SIZE)
    network = LeNet().to(device)
    opt = optim.SGD(network.parameters(), lr=LR)
    for i in range(EPOCH):
        lossEpoch = train(network, opt, dataloaderMNIST)
        wandb.log({"loss": lossEpoch})
        wandb.watch(network)
        if lossEpoch < modelLoss:
            modelLoss = lossEpoch
            torch.save(network.state_dict(), "./models/best.pt")
            print(f"epoch:{i}  loss:{lossEpoch}  model saved")
            continue
        print(f"epoch:{i}  loss:{lossEpoch}")

