import struct

import numpy as np


class READ:

    def __init__(self, imageFilePath, labelLabelPath):
        self.images = open(imageFilePath, "rb")
        self.labels = open(labelLabelPath, "rb")
        self.imgMagicNumber, self.imgNums = struct.unpack(">ii", self.images.read(8))
        self.imgWidth, self.imgHeight = struct.unpack(">ii", self.images.read(8))
        self.lblMagicNumber, self.lblNums = struct.unpack(">ii", self.labels.read(8))
        self.iterNums = self.imgNums

    def __iter__(self):
        return self

    def __next__(self):
        self.iterNums -= 1
        if self.iterNums > 0:
            image = [[struct.unpack(">B", self.images.read(1))[0] for j in range(0, 28)] for i in range(0, 28)]
            label = struct.unpack(">B", self.labels.read(1))
            return image, label
        else:
            raise StopIteration()

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


if __name__ == "__main__":
    pass
    # dataset = Dataset("mnistData/t10k-images.idx3-ubyte", "mnistData/t10k-labels.idx1-ubyte")
    # for img, lbl in dataset:
    #     image = np.array(img, dtype=np.uint8)
    #     cv2.imshow(str(lbl[0]), cv2.resize(image, (400, 400)))
    #     cv2.waitKey(80)
    # trainDataset = readMinistImages("./mnistData/train-images.idx3-ubyte")
    # testDataset = readMinistLabel()
    # for i in readMinistImages("./mnistData/train-images.idx3-ubyte"):
    #     cv2.imshow("test", i)
    #     cv2.waitKey(40)
    # for label, image in zip(readMinistLabel("mnistData/t10k-labels.idx1-ubyte"), readMinistImages(
    #         "mnistData/t10k-images.idx3-ubyte")):
    #     print(label)
    #     cv2.imwrite(f"./images/{label}.png", image)
    #     cv2.imshow("image", image)
    #     cv2.waitKey(0)
