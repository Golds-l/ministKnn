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


if __name__ == "__main__":
    trainImg = readMinistImages("./ministData/train-images.idx3-ubyte")
    testImg = readMinistImages("./ministData/t10k-images.idx3-ubyte")
    trainLbl = readMinistLabel("./ministData/train-labels.idx1-ubyte")
    testLbl = readMinistLabel("./ministData/t10k-labels.idx1-ubyte")
    print(trainImg.shape, trainLbl.shape)
    print(testImg.shape, testLbl.shape)
    num = 0
    for img, lbl in tqdm(zip(testImg[:100], testLbl[:100])):
        distanceAndLbl = [float('inf'), 0, 0]
        for trNum in range(len(trainImg)):
            # cv2.imshow("re", cv2.resize(img, (640, 640)))
            # distance = cv2.absdiff(img, trainImg[trNum])
            trLbl = trainLbl[trNum]
            distance = sum(sum([abs(i - j) for i, j in zip(img, trainImg[trNum])]))
            distanceAndLbl = [distance, trLbl, trainImg[trNum].copy()] if distance < distanceAndLbl[0] else [distanceAndLbl[0], distanceAndLbl[1], distanceAndLbl[2]]
            # cv2.imshow("training", cv2.resize(trainImg[trNum], (640, 640)))
            # cv2.imshow("nearestBefore", cv2.resize(distanceAndLbl[2], (640, 640)))
            # print(distanceAndLbl[:2])
            # if trLbl == 7:
            #     print(distance)
            #     cv2.waitKey(0)
            # cv2.waitKey(40)
        # print(distanceAndLbl[:2], lbl)
        if distanceAndLbl[1] == lbl:
            num += 1
        # cv2.imshow("near", cv2.resize(distanceAndLbl[2], (640, 640)))
        # cv2.imshow("origin", cv2.resize(img, (640, 640)))
        # cv2.waitKey(400)
    print(num / 100)

