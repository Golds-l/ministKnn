import struct
import time
from multiprocessing import Process, Queue

import numpy as np


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


def test(testImage, testLabel, trainImage, trainLabel, beginIndex, endIndex, que):
    num = 0
    for img, lbl in zip(testImage[beginIndex:endIndex], testLabel[beginIndex:endIndex]):
        distanceAndLbl = [float('inf'), 0, 0]
        for trImg, trLbl in zip(trainImage, trainLabel):
            distance = calculateDistance(img, trImg)
            distanceAndLbl = [distance, trLbl, trImg.copy()] if distance < distanceAndLbl[0] else [distanceAndLbl[0],
                                                                                                   distanceAndLbl[1],
                                                                                                   distanceAndLbl[2]]
        if distanceAndLbl[1] == lbl:
            num += 1
    que.put(num)
    return num / (endIndex - beginIndex)


def mulProcessTest(numOfProcess, que):
    size = int(10000 / numOfProcess)
    process = []
    for i in range(numOfProcess):
        process.append(Process(target=test, args=(testImg, testLbl, trainImg, trainLbl, i * size, i * size + size, que)))
    [p.start() for p in process]
    [p.join() for p in process]


if __name__ == "__main__":
    NUM_OF_PROCESS = 28
    nums = Queue(NUM_OF_PROCESS)
    timeBegin = time.time()
    trainImg = readMinistImages("./ministData/train-images.idx3-ubyte")
    testImg = readMinistImages("./ministData/t10k-images.idx3-ubyte")
    trainLbl = readMinistLabel("./ministData/train-labels.idx1-ubyte")
    testLbl = readMinistLabel("./ministData/t10k-labels.idx1-ubyte")
    mulProcessTest(NUM_OF_PROCESS, nums)
    print(f"time:{round(time.time() - timeBegin, 2)}s\taccuracy:{sum(nums.get() for i in range(NUM_OF_PROCESS))/10000}")
