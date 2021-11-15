import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

import wandb

EPOCH = 20
LR = 0.0004
BATCH_SIZE = 8


class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.FC1 = nn.Sequential(
            nn.Linear(28 * 28 * 1, 512),
            nn.ReLU()
        )
        self.FC2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.FC3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.FC4 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.FC5 = nn.Sequential(
            nn.Linear(32, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        x = self.FC4(x)
        x = self.FC5(x)
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
    wandb.init(project="FCNN")
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
    network = FCNN().to(device)
    opt = optim.SGD(network.parameters(), lr=LR)
    for i in range(EPOCH):
        lossEpoch = train(network, opt, dataloaderMNIST)
        wandb.log({"loss": lossEpoch})
        wandb.watch(network)
        if lossEpoch < modelLoss:
            modelLoss = lossEpoch
            torch.save(network.state_dict(), "./models/FCNN/best.pt")
            print(f"epoch:{i}  loss:{lossEpoch}  model saved!")
            continue
        print(f"epoch:{i}  loss:{lossEpoch}")
