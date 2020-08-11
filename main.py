import naiveBayes
import perceptron
import numpy as np
import util
import os
import random
import time

DIGIT_PIC_WIDTH = 28
DIGIT_PIC_HEIGHT = 28
FACE_PIC_WIDTH = 60
FACE_PIC_HEIGHT = 70


def basicFeatureExtractionDigit(pic: util.Picture):
    # a = pic.getPixels()

    features = util.Counter()

    for x in range(DIGIT_PIC_WIDTH):
        for y in range(DIGIT_PIC_HEIGHT):
            if pic.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


def basicFeatureExtractionFace(pic: util.Picture):
    # a = pic.getPixels()

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
    classifierType = "naiveBayes"
    # classifierType = "perceptron"

    # dataType = "digit"
    # legalLabels = range(10)
    dataType = "face"
    legalLabels = range(2)

    TRAINING_DATA_USAGE_SET = [round(i * 0.1, 1) for i in range(1, 11)]
    MAX_ITERATIONS = 10
    RANDOM_ITERATION = 1
    isTrainComplete = False

    TestDataIndex = [1, 2, 3]

    if os.path.exists('result') is False:
        os.mkdir('result')
    if os.path.exists('result/%s' % dataType) is False:
        os.mkdir('result/%s' % dataType)
    if os.path.exists('result/%s/%s' % (dataType, classifierType)) is False:
        os.mkdir('result/%s/%s' % (dataType, classifierType))
    resultStatisticFilePath = "result/%s/%s/StatisticData.txt" % (dataType, classifierType)
    resultWeightsFilePath = "result/%s/%s/WeightsData.txt" % (dataType, classifierType)
    resultWeightsGraphFilePath = "result/%s/%s/WeightGraph.txt" % (dataType, classifierType)

    if os.path.exists(resultWeightsFilePath):
        isTrainComplete = True
        # os.remove(resultWeightsFilePath)
    # if os.path.exists(resultStatisticFilePath):
    #     os.remove(resultStatisticFilePath)
    # if os.path.exists(resultWeightsGraphFilePath):
    #     os.remove(resultWeightsGraphFilePath)

    classifier = None
    if classifierType == "naiveBayes":
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
        print("Classifier Type: \033[1;32mNaive Bayes\033[0m")
    else:
        classifier = perceptron.PerceptronClassifier(legalLabels, MAX_ITERATIONS)
        print("Classifier Type: \033[1;32mPerceptron\033[0m")
        if isTrainComplete is True:
            print("\033[1;32mWeight File Detected!\033[0m The system will skip the training process and use the existed weight data.")
        else:
            print("\033[1;33mWeight File Not Existed!\033[0m The system will train the data to get the weight.")

    # classifier = perceptron.PerceptronClassifier(legalLabels, MAX_ITERATIONS)
    # print(classifier.weights)
    for TRAINING_DATA_USAGE in TRAINING_DATA_USAGE_SET:
        accuracy = []
        statisticResult = ""
        for randomTime in range(RANDOM_ITERATION):
            trainingData = None
            trainingLabels = None
            validationData = None
            validationLabels = None
            testData = None
            testLabels = None
            if dataType == "digit":
                TRAINING_SET_SIZE = int(
                    len(open("data/%sdata/traininglabels" % dataType, "r").readlines()) * TRAINING_DATA_USAGE)
                VALIDATION_SET_SIZE = int(len(open("data/%sdata/validationlabels" % dataType, "r").readlines()))
                if len(TestDataIndex) == 0:
                    TEST_SET_SIZE = int(len(open("data/%sdata/testlabels" % dataType, "r").readlines()))
                else:
                    TEST_SET_SIZE = len(TestDataIndex)
                print("Training Data Usage: %.1f%%" % (TRAINING_DATA_USAGE * 100))
                print("Random Time: %d" % randomTime)
                print("Training Set Size: %d" % TRAINING_SET_SIZE)
                print("Validation Set Size: %d" % VALIDATION_SET_SIZE)
                print("Test Set Size: %d" % TEST_SET_SIZE)

                randomOrder = random.sample(range(len(open("data/%sdata/traininglabels" % dataType, "r").readlines())),
                                            TRAINING_SET_SIZE)

                rawTrainingData = util.loadDataFileRandomly("data/%sdata/trainingimages" % dataType, randomOrder,
                                                            DIGIT_PIC_WIDTH, DIGIT_PIC_HEIGHT)
                trainingLabels = util.loadLabelFileRandomly("data/%sdata/traininglabels" % dataType, randomOrder)
                # print(len(rawTrainingData))

                rawValidationData = util.loadDataFile("data/%sdata/validationimages" % dataType, VALIDATION_SET_SIZE,
                                                      DIGIT_PIC_WIDTH, DIGIT_PIC_HEIGHT)
                validationLabels = util.loadLabelFile("data/%sdata/validationlabels" % dataType, VALIDATION_SET_SIZE)
                # print(len(rawValidationData))

                if len(TestDataIndex) == 0:
                    rawTestData = util.loadDataFile("data/%sdata/testimages" % dataType, TEST_SET_SIZE, DIGIT_PIC_WIDTH, DIGIT_PIC_HEIGHT)
                    testLabels = util.loadLabelFile("data/%sdata/testlabels" % dataType, TEST_SET_SIZE)
                    # print(len(rawTestData))
                else:
                    rawTestData = util.loadDataFileRandomly("data/%sdata/testimages" % dataType, TestDataIndex, DIGIT_PIC_WIDTH, DIGIT_PIC_HEIGHT)
                    testLabels = util.loadLabelFileRandomly("data/%sdata/testlabels" % dataType, TestDataIndex)

                print("\tExtracting features...", end="")
                trainingData = list(map(basicFeatureExtractionDigit, rawTrainingData))
                validationData = list(map(basicFeatureExtractionDigit, rawValidationData))
                testData = list(map(basicFeatureExtractionDigit, rawTestData))
                print("\033[1;32mDone!\033[0m")

            elif dataType == "face":
                TRAINING_SET_SIZE = int(len(open("data/%sdata/%sdatatrainlabels" % (dataType, dataType),
                                                 "r").readlines()) * TRAINING_DATA_USAGE)
                VALIDATION_SET_SIZE = int(
                    len(open("data/%sdata/%sdatavalidationlabels" % (dataType, dataType), "r").readlines()))
                TEST_SET_SIZE = int(len(open("data/%sdata/%sdatatestlabels" % (dataType, dataType), "r").readlines()))
                print("Training Data Usage: %.1f%%" % (TRAINING_DATA_USAGE * 100))
                print("Random Time: %d" % randomTime)
                print("Training Set Size: %d" % TRAINING_SET_SIZE)
                print("Validation Set Size: %d" % VALIDATION_SET_SIZE)
                print("Test Set Size: %d" % TEST_SET_SIZE)

                randomOrder = random.sample(
                    range(len(open("data/%sdata/%sdatatrainlabels" % (dataType, dataType), "r").readlines())),
                    TRAINING_SET_SIZE)
                # randomOrder = [i for i in range(TRAINING_SET_SIZE)]

                rawTrainingData = util.loadDataFileRandomly("data/%sdata/%sdatatrain" % (dataType, dataType),
                                                            randomOrder, FACE_PIC_WIDTH, FACE_PIC_HEIGHT)
                trainingLabels = util.loadLabelFileRandomly("data/%sdata/%sdatatrainlabels" % (dataType, dataType),
                                                            randomOrder)
                # print(len(rawTrainingData))

                rawValidationData = util.loadDataFile("data/%sdata/%sdatavalidation" % (dataType, dataType),
                                                      VALIDATION_SET_SIZE, FACE_PIC_WIDTH, FACE_PIC_HEIGHT)
                validationLabels = util.loadLabelFile("data/%sdata/%sdatavalidationlabels" % (dataType, dataType),
                                                      VALIDATION_SET_SIZE)
                # print(len(rawValidationData))

                if len(TestDataIndex) == 0:
                    rawTestData = util.loadDataFile("data/%sdata/%sdatatest" % (dataType, dataType), TEST_SET_SIZE, FACE_PIC_WIDTH, FACE_PIC_HEIGHT)
                    testLabels = util.loadLabelFile("data/%sdata/%sdatatestlabels" % (dataType, dataType), TEST_SET_SIZE)
                    # print(len(rawTestData))
                else:
                    rawTestData = util.loadDataFileRandomly("data/%sdata/%sdatatest" % (dataType, dataType), TestDataIndex, FACE_PIC_WIDTH, FACE_PIC_HEIGHT)
                    testLabels = util.loadLabelFileRandomly("data/%sdata/%sdatatestlabels" % (dataType, dataType), TestDataIndex)
                    # print(testLabels)

                print("\tExtracting features...", end="")
                trainingData = list(map(basicFeatureExtractionFace, rawTrainingData))
                validationData = list(map(basicFeatureExtractionFace, rawValidationData))
                testData = list(map(basicFeatureExtractionFace, rawTestData))
                print("\033[1;32mDone!\033[0m")

            statisticResult += "Training Data Usage: %.1f%%\tRandom Time: %d\n" % (TRAINING_DATA_USAGE * 100, randomTime)

            if (classifierType == "perceptron") and (isTrainComplete is True):
                print("\tLoading existing weight data...", end="")
                resultWeightsFile = open(resultWeightsFilePath, "r")
                index = int((TRAINING_DATA_USAGE * 10 - 1)) * 5 + randomTime

                for i in range(index):
                    resultWeightsFile.readline()
                classifier.weights = eval(resultWeightsFile.readline())
                for label, counter in classifier.weights.items():
                    Counter = util.Counter()
                    for key, value in counter.items():
                        Counter[key] = value
                    classifier.weights[label] = Counter
                # print(classifier.weights)
                # exit(1)
                print("\033[1;32mDone!\033[0m")
            else:
                print("\tTraining...")
                startTime = time.time()
                classifier.train(trainingData, trainingLabels, validationData, validationLabels)
                endTime = time.time()
                print("\t\033[1;32mTraining completed!\033[0m")
                print("\tTraining Time: \033[1;32m%.2f s\033[0m" % (endTime - startTime))

                statisticResult += "\tTraining Time: %.2f s\n" % (endTime - startTime)

            print("\tValidating...", end="")
            guesses = classifier.classify(validationData)
            correct = [guesses[i] == int(validationLabels[i]) for i in range(len(validationLabels))].count(True)
            print("\033[1;32mDone!\033[0m")
            print("\t\t", str(correct),
                  ("correct out of " + str(len(validationLabels)) + " (\033[1;32m%.2f%%\033[0m).") % (
                              100.0 * correct / len(validationLabels)))
            statisticResult += "\tValidation Accuracy: %s correct out of %s (%.2f%%)\n" % (str(correct), str(len(validationLabels)), (100.0 * correct / len(validationLabels)))

            print("\tTesting...", end="")
            guesses = classifier.classify(testData)
            correct = [guesses[i] == int(testLabels[i]) for i in range(len(testLabels))].count(True)
            print("\033[1;32mDone!\033[0m")
            print("\t\t", str(correct), ("correct out of " + str(len(testLabels)) + " (\033[1;32m%.2f%%\033[0m).") % (
                        100.0 * correct / len(testLabels)))
            statisticResult += "\tTest Accuracy: %s correct out of %s (%.2f%%)\n" % (str(correct), str(len(testLabels)), (100.0 * correct / len(testLabels)))
            accuracy.append(round(correct / len(testLabels), 4))
            if len(TestDataIndex) != 0:
                print("\t\tTest Data Predicted Label: %s" % guesses)
                print("\t\tTest Data Actual Label: %s" % list(int(i) for i in testLabels))

            if (classifierType == "perceptron") and (isTrainComplete is False):
                with open(resultWeightsFilePath, "a") as resultWeightsFile:
                    resultWeightsFile.write("%s\n" % str(classifier.weights))
            print()

            if (dataType == "digit") and (classifierType == "perceptron") and (isTrainComplete is False):
                weightPixels = ""
                for i in range(len(classifier.legalLabels)):
                    weightMatrix = np.zeros((DIGIT_PIC_WIDTH, DIGIT_PIC_HEIGHT))
                    for x, y in classifier.findHighWeightFeatures(int(classifier.legalLabels[i]),
                                                                  int(DIGIT_PIC_HEIGHT * DIGIT_PIC_WIDTH / 10)):
                        # print(x, y)
                        weightMatrix[x][y] = 1
                    # print(classifier.legalLabels[i])
                    weightPixels += "Training Data Usage: %.1f%%\tRandom Time: %d\tDigit: %s\n" % (
                    TRAINING_DATA_USAGE * 100, randomTime, classifier.legalLabels[i])
                    weightMatrix = np.rot90(weightMatrix, 1)
                    # np.flipud(weightMatrix)
                    for line in weightMatrix:
                        for character in line:
                            if int(character) == 0:
                                # print(" ", end="")
                                weightPixels += " "
                            else:
                                # print("#", end="")
                                weightPixels += "#"
                        # print()
                        weightPixels += "\n"
                with open(resultWeightsGraphFilePath, "a") as resultWeightsGraphFile:
                    resultWeightsGraphFile.write("%s\n" % weightPixels)
            elif (dataType == "face") and (classifierType == "perceptron") and (isTrainComplete is False):
                weightPixels = ""
                weightMatrix = np.zeros((FACE_PIC_WIDTH, FACE_PIC_HEIGHT))
                for x, y in classifier.findHighWeightFeatures(int(classifier.legalLabels[1]),
                                                              int(FACE_PIC_WIDTH * FACE_PIC_HEIGHT / 8)):
                    weightMatrix[x][y] = 1
                weightPixels += "Training Data Usage: %.1f%%\tRandom Time: %d\n" % (
                TRAINING_DATA_USAGE * 100, randomTime)
                weightMatrix = np.rot90(weightMatrix, 1)
                for line in weightMatrix:
                    for character in line:
                        if int(character) == 0:
                            # print(" ", end="")
                            weightPixels += " "
                        else:
                            # print("#", end="")
                            weightPixels += "#"
                    # print()
                    weightPixels += "\n"
                with open(resultWeightsGraphFilePath, "a") as resultWeightsGraphFile:
                    resultWeightsGraphFile.write("%s\n" % weightPixels)

        accuracyMean = np.mean(accuracy)
        accuracyStd = np.std(accuracy)
        print("Accuracy: ", accuracy)
        print("Accuracy Mean: \033[1;32m%.2f%%\033[0m" % (accuracyMean * 100))
        statisticResult += "Accuracy Mean: %.2f%%\t" % (accuracyMean * 100)
        print("Accuracy Standard Deviation: \033[1;32m%.2f%%\033[0m" % (accuracyStd * 100))
        statisticResult += "Accuracy Standard Deviation: %.8f\n" % accuracyStd
        if isTrainComplete is False:
            with open(resultStatisticFilePath, "a") as resultStatisticFile:
                resultStatisticFile.write(statisticResult)
        print()
