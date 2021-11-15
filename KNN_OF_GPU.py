# slowly!!!!!
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    trainMNIST = torchvision.datasets.MNIST(root="./mnistData", train=True, download=True, transform=transforms)
    testMNIST = torchvision.datasets.MNIST(root="./mnistData", train=False, download=True, transform=transforms)
    trainLoader = DataLoader(trainMNIST, batch_size=1)
    testLoader = DataLoader(testMNIST, batch_size=1)
    NUM_OF_RIGHT = torch.tensor(11)
    tB = time.time()
    for input, label in testLoader:
        input, label = input.cuda(), label.cuda()
        dis = torch.tensor(1000000, dtype=torch.float32)
        pred = torch.tensor(11)
        for trInput, trLbl in tqdm(trainLoader):
            trInput, trLbl = trInput.cuda(), trLbl.cuda()
            diff = input[0][0] - trInput[0][0]
            distance = diff.clamp(min=0, max=255).sum()
            if distance < dis:
                dis = distance
                pred = trLbl
        if pred == label:
            NUM_OF_RIGHT += 1
    print(f"num of right:{NUM_OF_RIGHT} time:{time.time() - tB}")
