import random
import util


class PerceptronClassifier:
    def __init__(self, legalLabels, maxIterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.maxIteration = maxIterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()

    def setWeight(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        learningRate = 1
        self.features = trainingData[0].keys()
        for label in self.legalLabels:
            self.weights[label][0] = 0.1
            for key in self.features:
                self.weights[label][key] = 0.5
            # print(self.weights[label])

        for iteration in range(self.maxIteration):
            print("Starting iteration %d..." % iteration, end="")
            i = 0
            allPassFlag = True
            while i < len(trainingData):
                # print("\tChecking Data %d..." % i, end="")
                result = {}
                for label in self.legalLabels:
                    result[label] = self.weights[label] * trainingData[i] + self.weights[label][0]

                isUpdate = False
                for key, value in result.items():
                    if value >= 0 and key != int(trainingLabels[i]):
                        # if isUpdate is False:
                        #     print("\033[1;31mError!\033[0m")
                        # print("\t\tUpdating weight %s..." % key, end="")
                        isUpdate = True
                        self.weights[key] = self.weights[key] - trainingData[i]
                        self.weights[key][0] = self.weights[key][0] + learningRate
                    elif value < 0 and key == int(trainingLabels[i]):
                        # if isUpdate is False:
                        #     print("\033[1;31mError!\033[0m")
                        # print("\t\tUpdating weight %s..." % key, end="")
                        isUpdate = True
                        self.weights[key] = self.weights[key] + trainingData[i]
                        self.weights[key][0] = self.weights[key][0] - learningRate
                if isUpdate is True:
                    allPassFlag = False
                    # print("%s" % result)
                    continue
                # else:
                #     print("\033[1;32mPass!\033[0m %s" % result)
                i += 1
            # print(self.weights)
            if allPassFlag is True:
                # print("\n\033[1;32mAll training data pass without any updates!\033[0m")
                # print(self.weights)
                print("\033[1;32mDone!\033[0m")
                break
            print("\033[1;32mDone!\033[0m")

    def classify(self, data):
        guesses = []
        for pic in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * pic + self.weights[l][0]
            guesses.append(vectors.argMax())
        return guesses

    def findHighWeightFeatures(self, label):
        featuresWeights = []
        return featuresWeights
