import naiveBayes
import perceptron
import numpy as np
import util
import sys

TEST_SET_SIZE = 100
DIGIT_PIC_WIDTH = 28
DIGIT_PIC_HEIGHT = 28
FACE_PIC_WIDTH = 60
FACE_PIC_HEIGHT = 70


def basicFeatureExtractionDigit(pic: util.Picture):
    a = pic.getPixels()

    features = util.Counter()

    for x in range(DIGIT_PIC_WIDTH):
        for y in range(DIGIT_PIC_HEIGHT):
            if pic.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


def basicFeatureExtractionFace(pic: util.Picture):
    a = pic.getPixels()

    features = util.Counter()

    for x in range(FACE_PIC_WIDTH):
        for y in range(FACE_PIC_HEIGHT):
            if pic.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


if __name__ == '__main__':
    np.set_printoptions(linewidth=400)
    # dataType = "digit"
    dataType = "face"
    TRAINING_DATA_USAGE = 0.1

    if dataType == "digit":
        TRAINING_SET_SIZE = int(len(open("data/%sdata/traininglabels" % dataType, "r").readlines()) * TRAINING_DATA_USAGE)
        print(TRAINING_SET_SIZE)

        rawTrainingData = util.loadDataFile("data/%sdata/trainingimages" % dataType, TRAINING_SET_SIZE, DIGIT_PIC_WIDTH, DIGIT_PIC_HEIGHT)
        trainingLabel = util.loadLabelFile("data/%sdata/traininglabels" % dataType, TRAINING_SET_SIZE)
        print(len(rawTrainingData))
        print(len(trainingLabel))
    elif dataType == "face":
        TRAINING_SET_SIZE = int(len(open("data/%sdata/facedatatrainlabels" % dataType, "r").readlines()) * TRAINING_DATA_USAGE)
        print(TRAINING_SET_SIZE)

        rawTrainingData = util.loadDataFile("data/%sdata/facedatatrain" % dataType, TRAINING_SET_SIZE, FACE_PIC_WIDTH,FACE_PIC_HEIGHT)
        trainingLabel = util.loadLabelFile("data/%sdata/facedatatrainlabels" % dataType, TRAINING_SET_SIZE)
        print(len(rawTrainingData))
        print(len(trainingLabel))
