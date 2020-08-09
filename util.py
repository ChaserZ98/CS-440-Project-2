import numpy as np
from functools import cmp_to_key


def IntegerConversionFunction(character):
    if character == ' ':
        return 0
    elif character == '+':
        return 1
    elif character == '#':
        return 2


def convertToInteger(data):
    if type(data) != np.ndarray:
        return IntegerConversionFunction(data)
    else:
        return np.array(list(map(convertToInteger, data)))


def AsciiGrayscaleConversionFunction(integer):
    if integer == 0:
        return ' '
    elif integer == 1:
        return '+'
    elif integer == 2:
        return '#'


def convertToAscii(data):
    if type(data) != np.ndarray:
        return AsciiGrayscaleConversionFunction(data)
    else:
        return np.array(list(map(convertToAscii, data)))


def loadDataFileRandomly(filepath: str, randomOrder: list, width: int, height: int):
    items = []
    file = open(filepath, "r")
    # print(randomOrder)
    for i in randomOrder:
        file.seek(i * (width + 1) * height, 0)
        # print("Data File i: ", i)
        # print("Data File: ", file.tell())
        picData = []
        for lineCounter in range(height):
            picData.append([character for character in file.readline() if character != '\n'])
        items.append(Picture(np.array(picData), width, height))
    return items


def loadLabelFileRandomly(filepath: str, randomOrder: list):
    file = open(filepath, "r")
    labels = []
    for i in randomOrder:
        file.seek(2 * i, 0)
        # print("Label File i: ", i)
        # print("Label File: ", file.tell())
        labels.append(file.read(1))
    return labels


def loadDataFile(filePath: str, totalPicNum: int, width: int, height: int):
    items = []
    file = open(filePath, "r")
    for i in range(totalPicNum):
        file.seek(i * (width + 1) * height, 0)
        picData = []
        for lineCounter in range(height):
            picData.append([character for character in file.readline() if character != '\n'])
        items.append(Picture(np.array(picData), width, height))
    return items


def loadLabelFile(filePath: str, totalPicNum: int):
    file = open(filePath, "r")
    labels = []
    for i in range(totalPicNum):
        file.seek(2 * i, 0)
        labels.append(file.read(1))
    return labels


class Picture:
    def __init__(self, data, width: int, height: int):
        self.width = width
        self.height = height
        if data is None:
            data = [[' ' for i in range(self.width)] for j in range(self.height)]
        self.pixels = np.rot90(convertToInteger(data), -1)

    def getPixel(self, column, row):
        return self.pixels[column][row]

    def getPixels(self):
        return self.pixels

    def getAsciiString(self):
        data = np.rot90(self.pixels, 1)
        ascii = convertToAscii(data)
        return '\n'.join(''.join(map(str, i)) for i in ascii)

    def __str__(self):
        return self.getAsciiString()


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


class Counter(dict):
    def __getitem__(self, index):
        self.setdefault(index, 0)
        return dict.__getitem__(self, index)

    def incrementALL(self, keys, count):
        for key in keys:
            self[key] += count

    def argMax(self):
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def sortedKeys(self):
        sortedItem = self.items()
        # print(sortedItem)
        # sortedItem.remove(1)
        # print(sortedItem)
        # compare = lambda x, y: sign(y[1] - x[1])
        sortedItem = sorted(sortedItem, key=lambda x: x[1], reverse=True)
        # print(sortedItem)
        # sortedItem.sort(cmp=compare)
        return [x[0] for x in sortedItem]

    def totalCount(self):
        return sum(self.values())

    def normalize(self):
        total = float(self.totalCount())
        if total == 0:
            return
        for key in self.keys():
            self[key] = self[key] / total

    def divideAll(self, divisor):
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        return Counter(dict.copy(self))

    def __mul__(self, y):
        sum = 0
        x = self
        if len(x) > len(y):
            x, y = y, x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum

    def __radd__(self, y):
        for key, value in y.items():
            self[key] += value

    def __add__(self, y):
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__(self, y):
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]
        return addend


if __name__ == '__main__':
    Width, Height = 60, 70
    dataPath = r'data/facedata/facedatatrain'
    labelPath = r'data/facedata/facedatatrainlabels'
    # Width, Height = 28, 28
    # dataPath = r'data/digitdata/trainingimages'
    # labelPath = r'data/digitdata/traininglabels'
    picture_index = 0

    pic = loadDataFile(dataPath, 4, Width, Height)
    # label = loadLabelFile(labelPath, i)
    # print("Label: %s" % label)
    print("Image: ")
    for i in range(len(pic)):
        print(pic[i])
        print("-------")
