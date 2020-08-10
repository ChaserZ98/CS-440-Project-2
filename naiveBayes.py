import random
import util
import math


class NaiveBayesClassifier:
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1  # this is the smoothing parameter, ** use it in your train method **
        # self.automaticTuning = False  # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([f for datum in trainingData for f in datum.keys()]));

        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]

        # if (self.automaticTuning):
        # kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        # else:
        # kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        result = []
        numberOfLabel = []
        numberOfSamples = len(trainingLabels)

        # total count of # in each feature
        countOfLabel = util.Counter()
        for key in trainingData[0]:
            countOfLabel[key] = 0

        for datum in trainingData:
            for key in datum:
                if datum[key] == 0:
                    countOfLabel[key] += 1

        # probability distribution of not empty for each feature
        for key in countOfLabel:
            countOfLabel[key] = countOfLabel[key] / numberOfSamples

        for label in self.legalLabels:
            result.append(util.Counter())
            numberOfLabel.append(0)
            for key in trainingData[0]:
                result[label][key] = 0

            for i in range(len(trainingLabels)):
                if int(trainingLabels[i]) == label:
                    numberOfLabel[label] += 1
                    for key in trainingData[i]:
                        if trainingData[i][key] == 0:
                            result[label][key] += 1

            # for key in result[int(label)]:
            # probability of empty space of each feature in  each label
            # result[int(label)][key] = result[int(label)][key]/numberOfLabel[int(label)]

        countOfValidation = len(validationLabels)
        # calculate probability of specific label
        pOfLabel = []
        for label in self.legalLabels:
            count = 0
            for i in range(len(validationLabels)):
                if int(validationLabels[i]) == label:
                    count += 1
            pOfLabel.append(count / len(validationLabels))

        bestK = 1
        bestValue = 0
        for i in range(len(kgrid)):
            correct = 0
            for j in range(len(validationLabels)):
                # convert label to integer
                realAnswer = int(validationLabels[j])
                probability = []
                for label in self.legalLabels:
                    logValue = math.log(pOfLabel[label])
                    # calculate conditional probability
                    for key in validationData[j]:
                        if validationData[j][key] == 0:
                            calculate1 = (result[label][key] + kgrid[i]) / (numberOfLabel[label] + 2 * kgrid[i])
                            logValue += math.log(calculate1)
                        else:
                            calculate2 = ((numberOfLabel[label] - result[label][key]) + kgrid[i]) / (
                                        numberOfLabel[label] + 2 * kgrid[i])
                            logValue += math.log(calculate2)
                    probability.append(logValue)
                # prediction of label
                answer = probability.index(max(probability))
                if answer == realAnswer:
                    correct += 1
            correct = correct / countOfValidation * 100
            if correct > bestValue:
                bestValue = correct
                bestK = kgrid[i]

        self.setSmoothing(bestK)
        self.result = result
        self.numberOfLabel = numberOfLabel
        self.pOfLabel = pOfLabel
        # print(bestValue)
        # print(bestK)

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = []  # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()

        "*** YOUR CODE HERE ***"
        for label in self.legalLabels:
            logValue = math.log(self.pOfLabel[label])
            # calculate conditional probability
            for key in datum:
                if datum[key] == 0:
                    calculate1 = (self.result[label][key] + self.k) / (self.numberOfLabel[label] + 2 * self.k)
                    logValue += math.log(calculate1)
                else:
                    calculate2 = ((self.numberOfLabel[label] - self.result[label][key]) + self.k) / (
                                self.numberOfLabel[label] + 2 * self.k)
                    logValue += math.log(calculate2)
            logJoint[label] = logValue
        # prediction of label
        return logJoint
