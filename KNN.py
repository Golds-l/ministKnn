import struct
import time
from multiprocessing import Process, Queue, Value

import numpy as np
import cv2


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
    return np.sum(cv2.absdiff(imgI, imgII))


def test(testImages, testLabels, trainImages, trainLabels, beginIndex, endIndex, val):  # 测试
    num = 0
    for img, lbl in zip(testImages[beginIndex:endIndex], testLabels[beginIndex:endIndex]):
        distanceAndLbl = [float('inf'), 0, 0]
        for trImg, trLbl in zip(trainImages, trainLabels):
            distance = calculateDistance(img, trImg)
            distanceAndLbl = [distance, trLbl, trImg.copy()] if distance < distanceAndLbl[0] else [distanceAndLbl[0],
                                                                                                   distanceAndLbl[1],
                                                                                                   distanceAndLbl[2]]
        if distanceAndLbl[1] == lbl:
            num += 1
    val.value += num
    return num / (endIndex - beginIndex)


def validation(image, trainImages, trainLabels):
    distanceAndLbl = [float('inf'), None, None]
    for trImg, trLbl in zip(trainImages, trainLabels):
        distance = calculateDistance(image, trImg)
        distanceAndLbl = [distance, trLbl, trImg.copy()] if distance < distanceAndLbl[0] else [distanceAndLbl[0],
                                                                                               distanceAndLbl[1],
                                                                                               distanceAndLbl[2]]
    return distanceAndLbl


def mulProcessTest(numOfProcess, val):
    size = int(10000 / numOfProcess)
    process = []
    for i in range(numOfProcess):
        process.append(
            Process(target=test, args=(testImg, testLbl, trainImg, trainLbl, i * size, i * size + size, val)))
    [p.start() for p in process]
    [p.join() for p in process]


if __name__ == "__main__":
    NUM_OF_PROCESS = 28
    numOfMismatch = 0
    numOfRight = Value("d", 0.0)
    trainImg = readMinistImages("mnistData/train-images.idx3-ubyte")
    testImg = readMinistImages("mnistData/t10k-images.idx3-ubyte")
    trainLbl = readMinistLabel("mnistData/train-labels.idx1-ubyte")
    testLbl = readMinistLabel("mnistData/t10k-labels.idx1-ubyte")
    timeBegin = time.time()
    mulProcessTest(NUM_OF_PROCESS, numOfRight)
    print(f"time: {round(time.time() - timeBegin, 2)}s\tnums of right: {numOfRight.value} accuracy: {numOfRight.value / 10000}")
