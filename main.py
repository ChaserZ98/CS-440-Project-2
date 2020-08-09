import naiveBayes
import perceptron
import numpy as np
import util
import os
import sys
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
    dataType = "digit"
    legalLabels = range(10)
    # dataType = "face"
    # legalLabels = range(2)
    TRAINING_DATA_USAGE_SET = [round(i*0.1, 1) for i in range(1, 11)]
    MAX_ITERATIONS = 10
    RANDOM_ITERATION = 5

    if os.path.exists('result') is False:
        os.mkdir('result')
    if os.path.exists('result/%s' % dataType) is False:
        os.mkdir('result/%s' % dataType)
    resultStatisticFilePath = "result/%s/StatisticData.txt" % dataType
    resultWeightsFilePath = "result/%s/WeightsData.txt" % dataType
    resultWeightsGraphFilePath = "result/%s/WeightGraph.txt" % dataType
    if os.path.exists(resultStatisticFilePath):
        os.remove(resultStatisticFilePath)
    if os.path.exists(resultWeightsFilePath):
        os.remove(resultWeightsFilePath)
    if os.path.exists(resultWeightsGraphFilePath):
        os.remove(resultWeightsGraphFilePath)

    classifier = perceptron.PerceptronClassifier(legalLabels, MAX_ITERATIONS)
    # print(classifier.weights)
    for TRAINING_DATA_USAGE in TRAINING_DATA_USAGE_SET:
        accuracy = []
        statisticResult = ""
        for randomTime in range(RANDOM_ITERATION):
            if dataType == "digit":
                TRAINING_SET_SIZE = int(len(open("data/%sdata/traininglabels" % dataType, "r").readlines()) * TRAINING_DATA_USAGE)
                VALIDATION_SET_SIZE = int(len(open("data/%sdata/validationlabels" % dataType, "r").readlines()))
                TEST_SET_SIZE = int(len(open("data/%sdata/testlabels" % dataType, "r").readlines()))
                print("Training Data Usage: %.1f%%" % (TRAINING_DATA_USAGE * 100))
                print("Random Time: %d" % randomTime)
                print("Training Set Size: %d" % TRAINING_SET_SIZE)
                print("Validation Set Size: %d" % VALIDATION_SET_SIZE)
                print("Test Set Size: %d" % TEST_SET_SIZE)

                randomOrder = random.sample(range(len(open("data/%sdata/traininglabels" % dataType, "r").readlines())), TRAINING_SET_SIZE)

                rawTrainingData = util.loadDataFileRandomly("data/%sdata/trainingimages" % dataType, randomOrder, DIGIT_PIC_WIDTH, DIGIT_PIC_HEIGHT)
                trainingLabels = util.loadLabelFileRandomly("data/%sdata/traininglabels" % dataType, randomOrder)
                # print(len(rawTrainingData))

                rawValidationData = util.loadDataFile("data/%sdata/validationimages" % dataType, VALIDATION_SET_SIZE,
                                                      DIGIT_PIC_WIDTH, DIGIT_PIC_HEIGHT)
                validationLabels = util.loadLabelFile("data/%sdata/validationlabels" % dataType, VALIDATION_SET_SIZE)
                # print(len(rawValidationData))

                rawTestData = util.loadDataFile("data/%sdata/testimages" % dataType, TEST_SET_SIZE, DIGIT_PIC_WIDTH,
                                                DIGIT_PIC_HEIGHT)
                testLabels = util.loadLabelFile("data/%sdata/testlabels" % dataType, TEST_SET_SIZE)
                # print(len(rawTestData))

                print("\tExtracting features...", end="")
                trainingData = list(map(basicFeatureExtractionDigit, rawTrainingData))
                validationData = list(map(basicFeatureExtractionDigit, rawValidationData))
                testData = list(map(basicFeatureExtractionDigit, rawTestData))
                print("\033[1;32mDone!\033[0m")

            elif dataType == "face":
                TRAINING_SET_SIZE = int(len(open("data/%sdata/%sdatatrainlabels" % (dataType, dataType), "r").readlines()) * TRAINING_DATA_USAGE)
                VALIDATION_SET_SIZE = int(
                    len(open("data/%sdata/%sdatavalidationlabels" % (dataType, dataType), "r").readlines()))
                TEST_SET_SIZE = int(len(open("data/%sdata/%sdatatestlabels" % (dataType, dataType), "r").readlines()))
                print("Training Data Usage: %.1f%%" % (TRAINING_DATA_USAGE * 100))
                print("Random Time: %d" % randomTime)
                print("Training Set Size: %d" % TRAINING_SET_SIZE)
                print("Validation Set Size: %d" % VALIDATION_SET_SIZE)
                print("Test Set Size: %d" % TEST_SET_SIZE)

                randomOrder = random.sample(range(len(open("data/%sdata/%sdatatrainlabels" % (dataType, dataType), "r").readlines())), TRAINING_SET_SIZE)
                # randomOrder = [i for i in range(TRAINING_SET_SIZE)]

                rawTrainingData = util.loadDataFileRandomly("data/%sdata/%sdatatrain" % (dataType, dataType), randomOrder, FACE_PIC_WIDTH, FACE_PIC_HEIGHT)
                trainingLabels = util.loadLabelFileRandomly("data/%sdata/%sdatatrainlabels" % (dataType, dataType), randomOrder)
                # print(len(rawTrainingData))

                rawValidationData = util.loadDataFile("data/%sdata/%sdatavalidation" % (dataType, dataType),
                                                      VALIDATION_SET_SIZE, FACE_PIC_WIDTH, FACE_PIC_HEIGHT)
                validationLabels = util.loadLabelFile("data/%sdata/%sdatavalidationlabels" % (dataType, dataType), VALIDATION_SET_SIZE)
                # print(len(rawValidationData))

                rawTestData = util.loadDataFile("data/%sdata/%sdatatest" % (dataType, dataType), TEST_SET_SIZE, FACE_PIC_WIDTH,
                                                FACE_PIC_HEIGHT)
                testLabels = util.loadLabelFile("data/%sdata/%sdatatestlabels" % (dataType, dataType), TEST_SET_SIZE)
                # print(len(rawTestData))

                print("\tExtracting features...", end="")
                trainingData = list(map(basicFeatureExtractionFace, rawTrainingData))
                validationData = list(map(basicFeatureExtractionFace, rawValidationData))
                testData = list(map(basicFeatureExtractionFace, rawTestData))
                print("\033[1;32mDone!\033[0m")

            statisticResult += "Training Data Usage: %.1f%%\tRandom Time: %d\n" % (TRAINING_DATA_USAGE * 100, randomTime)

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
            print("\t\t", str(correct), ("correct out of " + str(len(validationLabels)) + " (\033[1;32m%.2f%%\033[0m).") % (100.0 * correct / len(validationLabels)))
            statisticResult += "\tValidation Accuracy: %s correct out of %s (%.2f%%)\n" % (str(correct), str(len(validationLabels)), (100.0 * correct/len(validationLabels)))

            print("\tTesting...", end="")
            guesses = classifier.classify(testData)
            correct = [guesses[i] == int(testLabels[i]) for i in range(len(testLabels))].count(True)
            print("\033[1;32mDone!\033[0m")
            print("\t\t", str(correct), ("correct out of " + str(len(testLabels)) + " (\033[1;32m%.2f%%\033[0m).") % (100.0 * correct / len(testLabels)))
            statisticResult += "\tTest Accuracy: %s correct out of %s (%.2f%%)\n" % (str(correct), str(len(testLabels)), (100.0 * correct/len(testLabels)))
            accuracy.append(round(correct/len(testLabels), 4))

            with open(resultWeightsFilePath, "a") as resultWeightsFile:
                resultWeightsFile.write("%s\n" % str(classifier.weights))
            print()

            if dataType == "digit":
                weightPixels = ""
                for i in range(len(classifier.legalLabels)):
                    weightMatrix = np.zeros((DIGIT_PIC_WIDTH, DIGIT_PIC_HEIGHT))
                    for x, y in classifier.findHighWeightFeatures(int(classifier.legalLabels[i]), int(DIGIT_PIC_HEIGHT * DIGIT_PIC_WIDTH / 10)):
                        # print(x, y)
                        weightMatrix[x][y] = 1
                    # print(classifier.legalLabels[i])
                    weightPixels += "Training Data Usage: %.1f%%\tRandom Time: %d\tDigit: %s\n" % (TRAINING_DATA_USAGE * 100, randomTime, classifier.legalLabels[i])
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
            elif dataType == "face":
                weightPixels = ""
                weightMatrix = np.zeros((FACE_PIC_WIDTH, FACE_PIC_HEIGHT))
                for x, y in classifier.findHighWeightFeatures(int(classifier.legalLabels[1]), int(FACE_PIC_WIDTH * FACE_PIC_HEIGHT / 8)):
                    weightMatrix[x][y] = 1
                weightPixels = "Training Data Usage: %.1f%%\tRandom Time: %d\n" % (TRAINING_DATA_USAGE * 100, randomTime)
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
        accuracyStd = np.std(accuracy, ddof=1)
        print("Accuracy: ", accuracy)
        print("Accuracy Mean: \033[1;32m%.2f%%\033[0m" % (accuracyMean * 100))
        statisticResult += "Accuracy Mean: %.2f%%\t" % (accuracyMean * 100)
        print("Accuracy Standard Deviation: \033[1;32m%.8f\033[0m" % accuracyStd)
        statisticResult += "Accuracy Standard Deviation: %.8f\n" % accuracyStd
        with open(resultStatisticFilePath, "a") as resultStatisticFile:
            resultStatisticFile.write(statisticResult)
        print()
