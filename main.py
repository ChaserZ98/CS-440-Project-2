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


if __name__ == '__main__':
    np.set_printoptions(linewidth=400)
    dataType = "digit"
    TRAINING_DATA_USAGE = 0.1

    if dataType == "digit":

        TRAINING_SET_SIZE = len(open("data/%sdata/traininglabels" % dataType, "r").readlines()) * TRAINING_DATA_USAGE
        print(TRAINING_SET_SIZE)

        rawTrainingData = util.loadDataFile("data/%sdata/trainingimages" % dataType, 0, DIGIT_PIC_WIDTH, DIGIT_PIC_HEIGHT)
        trainingLabel = util.loadLabelFile("data/%sdata/traininglabels" % dataType, 0)
        print(rawTrainingData)
        print(trainingLabel)