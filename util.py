import numpy as np


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


def loadDataFile(filePath: str, pictureIndex: int, width: int, height: int):
    file = open(filePath, "r")
    file.seek(pictureIndex * (width + 1) * height, 0)
    result = []
    for lineCounter in range(height):
        result.append([character for character in file.readline() if character != '\n'])
    return np.array(result)


def loadLabelFile(filePath: str, pictureIndex: int):
    file = open(filePath, "r")
    file.seek(2 * pictureIndex, 0)
    return file.read(1)


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


if __name__ == '__main__':

    Width, Height = 60, 70
    dataPath = r'data/facedata/facedatatrain'
    labelPath = r'data/facedata/facedatatrainlabels'
    # Width, Height = 28, 28
    # dataPath = r'data/digitdata/trainingimages'
    # labelPath = r'data/digitdata/traininglabels'
    picture_index = 0
    np.set_printoptions(linewidth=400)
    for i in range(4):
        pic = Picture(loadDataFile(dataPath, i, Width, Height), Width, Width)
        label = loadLabelFile(labelPath, i)
        print("Label: %s" % label)
        print("Image: ")
        print(pic)

