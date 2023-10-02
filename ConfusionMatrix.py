class confusion_matrix:
    def __init__(self, matrix=[]):
        self.matrix = matrix

    def print(self):
        print("confusion matrix: ")
        print(self.matrix)

        print("accuracy: ", self.accuracy())

        print("precision: ", self.precision())

        print("recall: ", self.recall())

        print("f1_score: ", self.f1_score())

    def fill(self, actual, predicted):
        self.matrix[actual][predicted] += 1

    # assuming the matrix is n*n

    def accuracy(self):
        total = 0

        correct = 0

        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                total += self.matrix[i][j]

                if i == j:
                    correct += self.matrix[i][j]

        return correct / total

    def precision(self):
        precision = []

        for i in range(len(self.matrix)):
            tp = self.matrix[i][i]

            fp = 0

            for j in range(len(self.matrix)):
                if i != j:
                    fp += self.matrix[j][i]

            precision.append(tp / (tp + fp))

        return precision

    def recall(self):
        recall = []

        for i in range(len(self.matrix)):
            tp = self.matrix[i][i]

            fn = 0

            for j in range(len(self.matrix)):
                if i != j:
                    fn += self.matrix[i][j]

            recall.append(tp / (tp + fn))

        return recall

    def f1_score(self):
        precision = self.precision()
        recall = self.recall()

        f1_score = []

        for i in range(len(self.matrix)):
            f1_score.append(2 * precision[i] * recall[i] / (precision[i] + recall[i]))

        return f1_score