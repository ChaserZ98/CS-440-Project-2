import naiveBayes
import perceptron
import numpy as np
import util
import sys
import random

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
    dataType = "digit"
    legalLabels = range(10)
    # dataType = "face"
    # legalLabels = range(2)
    TRAINING_DATA_USAGE = 0.1
    MAX_ITERATIONS = 10

    # w = util.Counter()
    # x = util.Counter()
    # w[0] = 1
    # w[1] = 2
    # w[2] = 3
    # x[1] = 1
    # x[2] = 2
    # print(x)
    # print(w)
    # print(w*x + w[0])
    # exit()

    classifier = perceptron.PerceptronClassifier(legalLabels, MAX_ITERATIONS)
    print(classifier.weights)

    if dataType == "digit":
        TRAINING_SET_SIZE = int(
            len(open("data/%sdata/traininglabels" % dataType, "r").readlines()) * TRAINING_DATA_USAGE)
        VALIDATION_SET_SIZE = int(len(open("data/%sdata/validationlabels" % dataType, "r").readlines()))
        TEST_SET_SIZE = int(len(open("data/%sdata/testlabels" % dataType, "r").readlines()))
        print("Training Set Size: %d" % TRAINING_SET_SIZE)
        print("Validation Set Size: %d" % VALIDATION_SET_SIZE)
        print("Test Set Size: %d" % TEST_SET_SIZE)

        rawTrainingData = util.loadDataFile("data/%sdata/trainingimages" % dataType, TRAINING_SET_SIZE, DIGIT_PIC_WIDTH,
                                            DIGIT_PIC_HEIGHT)
        trainingLabels = util.loadLabelFile("data/%sdata/traininglabels" % dataType, TRAINING_SET_SIZE)
        # print(len(rawTrainingData))

        rawValidationData = util.loadDataFile("data/%sdata/validationimages" % dataType, VALIDATION_SET_SIZE,
                                              DIGIT_PIC_WIDTH, DIGIT_PIC_HEIGHT)
        validationLabels = util.loadLabelFile("data/%sdata/validationlabels" % dataType, VALIDATION_SET_SIZE)
        # print(len(rawValidationData))

        rawTestData = util.loadDataFile("data/%sdata/testimages" % dataType, TEST_SET_SIZE, DIGIT_PIC_WIDTH,
                                        DIGIT_PIC_HEIGHT)
        testLabels = util.loadLabelFile("data/%sdata/testlabels" % dataType, TEST_SET_SIZE)
        # print(len(rawTestData))

        print("Extracting features...", end="")
        trainingData = list(map(basicFeatureExtractionDigit, rawTrainingData))
        validationData = list(map(basicFeatureExtractionDigit, rawValidationData))
        testData = list(map(basicFeatureExtractionDigit, rawTestData))
        print("done!")

        # temp = basicFeatureExtractionDigit(rawTrainingData[0])
        # counter = 0
        # weights = {}
        # weights[1] = util.Counter()
        # for key in trainingData[0].keys():
        #     weights[1][key] = 0.5
        #     # counter += 1
        # print(trainingData[0])

    elif dataType == "face":
        TRAINING_SET_SIZE = int(
            len(open("data/%sdata/%sdatatrainlabels" % (dataType, dataType), "r").readlines()) * TRAINING_DATA_USAGE)
        VALIDATION_SET_SIZE = int(
            len(open("data/%sdata/%sdatavalidationlabels" % (dataType, dataType), "r").readlines()))
        TEST_SET_SIZE = int(len(open("data/%sdata/%sdatatestlabels" % (dataType, dataType), "r").readlines()))
        print("Training Set Size: %d" % TRAINING_SET_SIZE)
        print("Validation Set Size: %d" % VALIDATION_SET_SIZE)
        print("Test Set Size: %d" % TEST_SET_SIZE)

        rawTrainingData = util.loadDataFile("data/%sdata/facedatatrain" % dataType, TRAINING_SET_SIZE, FACE_PIC_WIDTH,
                                            FACE_PIC_HEIGHT)
        trainingLabels = util.loadLabelFile("data/%sdata/facedatatrainlabels" % dataType, TRAINING_SET_SIZE)
        print(len(rawTrainingData))

        rawValidationData = util.loadDataFile("data/%sdata/%sdatavalidation" % (dataType, dataType),
                                              VALIDATION_SET_SIZE, FACE_PIC_WIDTH, FACE_PIC_HEIGHT)
        validationLabels = util.loadLabelFile("data/%sdata/%sdatavalidationlabels" % (dataType, dataType),
                                              VALIDATION_SET_SIZE)
        print(len(rawValidationData))

        rawTestData = util.loadDataFile("data/%sdata/%sdatatest" % (dataType, dataType), TEST_SET_SIZE, FACE_PIC_WIDTH,
                                        FACE_PIC_HEIGHT)
        testLabels = util.loadLabelFile("data/%sdata/%sdatatestlabels" % (dataType, dataType), TEST_SET_SIZE)
        print(len(rawTestData))

        print("Extracting features...")
        trainingData = map(basicFeatureExtractionFace, rawTrainingData)
        validationData = map(basicFeatureExtractionFace, rawValidationData)
        testData = map(basicFeatureExtractionFace, rawTestData)

    print("Training...")
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print("done!\n")

    print("Validating...", end="")
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == int(validationLabels[i]) for i in range(len(validationLabels))].count(True)
    print("done!")
    print("\t", str(correct), ("correct out of " + str(len(validationLabels)) + " (%.2f%%).") % (100.0 * correct / len(validationLabels)))

    print("Testing...", end="")
    guesses = classifier.classify(testData)
    correct = [guesses[i] == int(testLabels[i]) for i in range(len(testLabels))].count(True)
    print("done!")
    print("\t", str(correct), ("correct out of " + str(len(testLabels)) + " (%.2f%%).") % (100.0 * correct / len(testLabels)))
    # analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

