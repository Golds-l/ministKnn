import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import time

from LeNet import LeNet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = LeNet().to(device)
    network.load_state_dict(torch.load("./models/best.pt"))

    transform = transforms.ToTensor()
    img = transform(Image.open("./images/9.png")).to(device)
    img = torch.unsqueeze(img, dim=0)

    tB = time.time()
    out = network(img)
    print(torch.argmax(out).item(), time.time() - tB)
