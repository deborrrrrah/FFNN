import random
import numpy as np
import math

class MiniBatch:
    # data structure
    # weights
    __errors = []
    __outputs = []
    __weights = []
    __weights_bef = []

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
        self.__weights_bef = []

        # number of nodes for each layer
        self.__n_nodes = [self.__n_features] + [self.__nb_nodes] * self.__hidden_layer + [1] # 1 for output

        for i in range (0, self.__hidden_layer - 1) :
            b = []
            b_bef = []
            for _ in range (0, self.__n_nodes[i] + 1) :
                # index 0 for bias
                c = []
                c_bef = []
                for _ in range (0, self.__n_nodes[i + 1]) :
                    c.append(random.random())
                    c_bef.append(0)
                b.append(c)
                b_bef.append(c_bef)
            self.__weights.append(b)
            self.__weights_bef.append(b_bef)
        
    def __sigmoid(self, v) :
        return 1/(1 + math.exp(-v))

    def __psi_apostrophe(self, value) :
        return value * (1 - value)
        
    def __generate_batch(self) :
        return 

    def __forward_pass(self) :
        self.__outputs = []
        
        for i in range (0, self.__hidden_layer) : # last index for output
            b = [] # for output in each layer
            for j in range (0, self.__n_nodes[i+1]) :
                v = 0 # output value before using activation function 
                for j in range (0, self.__n_nodes[i]) :
                    v = v + self.__weights[i][j][i]

    def __backward_pass(self, y_batch) :

        # initialize errors to all zero
        self.__errors = []
        
        for i in range (0, self.__hidden_layer) :
            self.__errors.append([0] * self.__nb_nodes)
        
        self.__errors.append([0])
        
        for idx, output in enumerate(self.__outputs) :
            temp_error = []

            # output layer
            o_idx = self.__hidden_layer
            o_predict = output[o_idx][0]
            temp_error.insert(0, [self.__psi_apostrophe(o_predict) * (y_batch[idx] - o_predict)])

            # hidden layer
            # -1 karena index dimulai dari 0
            for i in range (len(output) - 1, -1, -1) :
                # perkalian matriks
                matrix_error = np.matrix(temp_error[0])
                matrix_weight = np.matrix(self.__weights[i])

                result = matrix_weight.dot(matrix_error.T) 
                result = np.squeeze(np.asarray(result.T)).tolist()

                del result[0]

                temp_error.insert(0, list(map(lambda x, y: self.__psi_apostrophe(x) + y, output[i], result)))

            # append output 
            for i in range (self.__errors) :
                self.__errors[i] = list(map(lambda x, y : x + y, self.__errors[i], temp_error[i]))

    def __update_weights(self) :
        return

    def fit(self, X, y) :
        # X is pandas.dataframe
        # y is pandas.series
        self.__X_train = X
        self.__y_train = y
        self.__random_weights()
        batches = self.__generate_batch()

        for batch in batches :
            self.__batch = batch
            self.__forward_pass()
            self.__backward_pass()
            self.__update_weights()

        return

    def predict(self, X) :
        # X is pandas.dataframe

        return
