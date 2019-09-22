import random

class MiniBatch:
    # data structure
    # weights
    __errors = []
    __outputs = []
    __weights = []

    def __init__(self, nb_nodes, hidden_layer, batch_size, learning_rate, momentum, epoch) :
        self.__nb_nodes = nb_nodes
        self.__hidden_layer = hidden_layer
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__epoch = epoch

    def __random_weights(self) :
        # bias in index 0
        self.__n_features = len(self.__X_train.columns)

        self.__weights = []

        weight = []
        for i in range (0, self.__n_features) :
            weight.append(random.random())
        self.__weights.append(weight)

        n_nodes = [self.__n_features, self.__nb_nodes * self.__hidden_layer, 1] # 1 for output

        for i in range (0, self.__hidden_layer - 1) :
            b = []
            for j in range (0, n_nodes[i] + 1) :
                # index 0 for bias
                c = []
                for k in range (0, n_nodes[i + 1]) :
                    c.append(random.random())
                b.append(c)
            self.__weights.append(b)

        for i in range (0, self.__hidden_layer) :
            weight = []
            for j in range (0, )

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