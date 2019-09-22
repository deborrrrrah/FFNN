import random

class MiniBatch:
    # data structure
    # weights

    def __init__(self, nb_nodes, hidden_layer, batch_size, learning_rate, momentum, epoch) :
        self.nb_nodes = nb_nodes
        self.hidden_layer = hidden_layer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epoch = epoch

    def __random_weights(self) :
        self.n_features = len(self.X_train.columns)

        self.weights = []

        weight = []
        for i in range (0, self.n_features) :
            weight.append(random.random())
        self.weights.append(weight)

        for i in range (0, self.hidden_layer) :
            weight = []
            for j in range (0, self.nb_nodes) :
                weight.append(random.random())
            self.weights.append(weight)

    def __forward_pass(self) :

    def __backward_pass(self) :

    def __update(self) :
        
    def __sigmoid(self) :

    def fit(self, X, y) :
        # X is pandas.dataframe
        # y is pandas.series
        self.__X_train = X
        self.__y_train = y
        self.__random_weights()

        return

    def predict(self, X) :
        # X is pandas.dataframe

        return