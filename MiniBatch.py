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
    __n_batches = 0
    __batch_X = []
    __batch_y = []

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
        total_row = len(self.__X_train)
        self.__n_batches = total_row/self.__batch_size

        indexes = [x for x in range(0, total_row)]

        random.shuffle(indexes)



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
            self.__errors.append(temp_error) 

    def __update_weights(self) :
        temp_weights = []

        # delta weight
        delta_weights = []
        for i in range(len(self._weights)) :
            delta_weight = []
            for j in range(len(self.__weights[i])):
                deltas = []
                for k in range(len(self.__weights[i][j])) :
                    delta = 0
                    for idx, output in enumerate(self.__outputs) :
                        output.insert(0, self.__batch_X.iloc[idx].tolist())
                        delta += (self.__momentum * self.__weights_bef[i][j][k]) + (self.__learning_rate * self.__errors[idx][i][k] * output[idx][i][j])
                    deltas.append(delta)
                delta_weight.append(deltas)
            delta_weights.append(delta_weights)

        # update weight
        for i in range(len(self.__weights)) :
            for j in range(len(self.__weights[i])) :
                for k in range(len(self.__weights[i][j])) :
                    temp_weights[i][j][k] = self.__weights[i][j][k] + delta_weights[i][j][k]

        self.__weights_bef = self.__weights
        self.__weights = temp_weights

    def fit(self, X, y) :
        # X is pandas.dataframe
        # y is pandas.series
        self.__X_train = X
        self.__y_train = y
        self.__random_weights()

        for i in range (self.__epoch) :
            self.__generate_batch()

            for i in range (self.__n_batches) :
                self.__batch_X = self.__batches_X[i]
                self.__batch_y = self.__batches_y[i]
                self.__forward_pass()
                self.__backward_pass()
                self.__update_weights()

        return

    def predict(self, X) :
        # X is pandas.dataframe

        return
