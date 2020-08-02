import util


class PerceptronClassifier:
    def __init__(self, actualLabels, maxIterations):
        self.actualLabels = actualLabels
        self.type = "perceptron"
        self.maxIteration = maxIterations
        self.weights = {}
