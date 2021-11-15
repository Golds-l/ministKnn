import torch
from torchvision import transforms
from PIL import Image
import time

from LeNet import LeNet
from ministRead import READ

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NETWORK = LeNet().to(DEVICE)
NETWORK.load_state_dict(torch.load("./models/LeNet/best.pt"))


def singlePred(imgPath):
    transform = transforms.ToTensor()
    image = transform(Image.open(imgPath)).to(DEVICE)
    imageTensor = torch.unsqueeze(image, dim=0)
    output = NETWORK(imageTensor)
    return torch.argmax(output).item()


if __name__ == "__main__":
    # print(singlePred("images/0.png"))
    mnist = READ("mnistData/t10k-images.idx3-ubyte", "mnistData/t10k-labels.idx1-ubyte")
    tB = time.time()
    NUM_OF_RIGHT = 0
    for img, lbl in mnist:
        imgTensor = torch.unsqueeze(torch.unsqueeze(torch.tensor(img, dtype=torch.float32).to(DEVICE), dim=0), dim=0)
        out = NETWORK(imgTensor)
        pred = torch.argmax(out).item()
        if lbl[0] == pred:
            NUM_OF_RIGHT += 1
        # print("true" if lbl[0] == pred else "false")
    print(time.time() - tB, NUM_OF_RIGHT)
    print(f"num of right:{NUM_OF_RIGHT}  time:{time.time() - tB} accuracy:{NUM_OF_RIGHT/mnist.imgNums}")
