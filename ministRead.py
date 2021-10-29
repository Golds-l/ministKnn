import struct
import time

import numpy as np
import cv2


class Dataset:
    def __init__(self, imageFilePath, labelLabelPath):
        self.imgPt = imageFilePath
        self.lblPt = labelLabelPath

    @staticmethod
    def readMinistImages(filePath):
        with open(filePath, "rb") as fp:
            magicNumber, sampleNums = struct.unpack(">ii", fp.read(8))
            rowsNums, columnsNums = struct.unpack(">ii", fp.read(8))
            images = np.fromfile(fp, dtype=np.uint8).reshape(sampleNums, rowsNums, columnsNums)
        return images

    @staticmethod
    def readMinistLabel(filePath):
        with open(filePath, "rb") as fp:
            magicNumber, sampleNums = struct.unpack(">ii", fp.read(8))
            labels = np.fromfile(fp, dtype=np.uint8)
        return labels


class TestIter:
    def __init__(self, i):
        self.i = i

    def __next__(self):
        self.i += 1
        return self.i

    def __iter__(self):
        return self


if __name__ == "__main__":
    test = TestIter(5)
    for i in test:
        print(i)
        time.sleep(1)
    # trainDataset = readMinistImages("./ministData/train-images.idx3-ubyte")
    # testDataset = readMinistLabel()
    # for i in readMinistImages("./ministData/train-images.idx3-ubyte"):
    #     cv2.imshow("test", i)
    #     cv2.waitKey(40)
    # for label, image in zip(readMinistLabel("ministData/t10k-labels.idx1-ubyte"), readMinistImages(
    #         "ministData/t10k-images.idx3-ubyte")):
    #     print(label)
    #     cv2.imwrite(f"./images/{label}.png", image)
    #     cv2.imshow("image", image)
    #     cv2.waitKey(0)
