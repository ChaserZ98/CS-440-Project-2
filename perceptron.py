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

        bestWeights = {}
        bestAccuracy = 0
        for iteration in range(self.maxIteration):
            print("\t\tStarting iteration %d..." % iteration, end="")
            i = 0
            allPassFlag = True
            while i < len(trainingData):
                # print("\tChecking Data %d..." % i, end="")
                result = {}
                for label in self.legalLabels:
                    result[label] = self.weights[label] * trainingData[i] + self.weights[label][0]

                isUpdate = False
                largestValue = max(result.values())
                predictionKey = None
                for key, value in result.items():
                    if value == largestValue:
                        predictionKey = key
                if predictionKey != int(trainingLabels[i]):
                    if result[predictionKey] > 0:
                        self.weights[predictionKey] = self.weights[predictionKey] - trainingData[i]
                        self.weights[predictionKey][0] = self.weights[predictionKey][0] - learningRate
                        isUpdate = True
                    if result[int(trainingLabels[i]) < 0]:
                        isUpdate = True
                        self.weights[int(trainingLabels[i])] = self.weights[int(trainingLabels[i])] + trainingData[i]
                        self.weights[predictionKey][0] = self.weights[predictionKey][0] + learningRate
                    # if value >= 0 and key != int(trainingLabels[i]):
                    #     # if isUpdate is False:
                    #     #     print("\033[1;31mError!\033[0m")
                    #     # print("\t\tUpdating weight %s..." % key, end="")
                    #     isUpdate = True
                    #     self.weights[key] = self.weights[key] - trainingData[i]
                    #     self.weights[key][0] = self.weights[key][0] + learningRate
                    # elif value < 0 and key == int(trainingLabels[i]):
                    #     # if isUpdate is False:
                    #     #     print("\033[1;31mError!\033[0m")
                    #     # print("\t\tUpdating weight %s..." % key, end="")
                    #     isUpdate = True
                    #     self.weights[key] = self.weights[key] + trainingData[i]
                    #     self.weights[key][0] = self.weights[key][0] - learningRate
                if isUpdate is True:
                    allPassFlag = False
                    # print("%s" % result)
                    # continue
                # else:
                #     print("\033[1;32mPass!\033[0m %s" % result)
                i += 1
            # print(self.weights)

            guesses = self.classify(validationData)
            correct = [guesses[i] == int(validationLabels[i]) for i in range(len(validationLabels))].count(True)
            accuracy = correct/len(validationLabels)

            if accuracy > bestAccuracy:
                bestWeights = self.weights
                bestAccuracy = accuracy

            if allPassFlag is True:
                # print("\n\033[1;32mAll training data pass without any updates!\033[0m")
                # print(self.weights)
                print("\033[1;32mDone!\033[0m")
                break
            print("\033[1;32mDone!\033[0m")
        self.weights = bestWeights

    def classify(self, data):
        guesses = []
        for pic in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * pic + self.weights[l][0]
            guesses.append(vectors.argMax())
        return guesses

    def findHighWeightFeatures(self, label, weightNum: int):
        featuresWeights = []
        sortedItems = self.weights[label].sortedKeys()
        # print(sortedItems)
        for i in range(1, weightNum):
            if type(sortedItems[i]) is tuple:
                featuresWeights.append(sortedItems[i])
        return featuresWeights
