import struct

import numpy as np
import cv2
from tqdm import tqdm


def readMinistLabel(filePath):
    with open(filePath, "rb") as fp:
        magicNumber, sampleNums = struct.unpack(">ii", fp.read(8))
        labels = np.fromfile(fp, dtype=np.uint8)
    return labels


def readMinistImages(filePath):
    with open(filePath, "rb") as fp:
        magicNumber, sampleNums = struct.unpack(">ii", fp.read(8))
        rowsNums, columnsNums = struct.unpack(">ii", fp.read(8))
        images = np.fromfile(fp, dtype=np.uint8).reshape(sampleNums, rowsNums, columnsNums)
    return images


def calculateDistance(imgI, imgII):
    return np.sum(imgII - imgI)


if __name__ == "__main__":
    trainImg = readMinistImages("./ministData/train-images.idx3-ubyte")
    testImg = readMinistImages("./ministData/t10k-images.idx3-ubyte")
    trainLbl = readMinistLabel("./ministData/train-labels.idx1-ubyte")
    testLbl = readMinistLabel("./ministData/t10k-labels.idx1-ubyte")
    num = 0
    for img, lbl in tqdm(zip(testImg, testLbl)):
        distanceAndLbl = [float('inf'), 0, 0]
        for trImg, trLbl in zip(trainImg, trainLbl):
            distance = calculateDistance(img, trImg)
            distanceAndLbl = [distance, trLbl, trImg.copy()] if distance < distanceAndLbl[0] else [distanceAndLbl[0], distanceAndLbl[1], distanceAndLbl[2]]
        if distanceAndLbl[1] == lbl:
            num += 1
    print(num / testImg.shape[0])

