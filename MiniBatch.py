import random
import numpy as np
import pandas as pd
import math

IntegerTypes = (int)
NumberTypes = (int, float)

class MiniBatch:
    # data structure
    # weights
    __errors = []
    __outputs = []
    __weights = []
    __weights_bef = []
    __batch_X = [] # pandas dataframe
    __batch_y = []

    def __init__(self, nb_nodes, hidden_layer, batch_size, learning_rate, momentum, epoch) :
        if not isinstance(nb_nodes, IntegerTypes) :
            raise TypeError('nb_nodes must be a integer')
        elif not isinstance(hidden_layer, IntegerTypes) :
            raise TypeError('hidden_layer must be a integer')
        elif not isinstance(learning_rate, NumberTypes) :
            raise TypeError('learning_rate must be a number')
        elif not isinstance(momentum, NumberTypes) :
            raise TypeError('momentum must be a number')
        elif not isinstance(epoch, IntegerTypes) :
            raise TypeError('epoch must be a number')
        elif learning_rate > 1 or learning_rate < 0 :
            raise ValueError('learning_rate must be between 0 and 1')
        elif momentum > 1 or momentum < 0 :
            raise ValueError('momentum must be between 0 and 1')
        elif hidden_layer > 10 or hidden_layer < 1 :
            raise ValueError('hidden_layer minimum 1 and maximum 10')

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

        for i in range (0, self.__hidden_layer + 1) :
            b = []
            b_bef = []
            for _ in range (0, self.__n_nodes[i] + 1) :
                # index 0 for bias
                c = []
                c_bef = []
                for _ in range (0, self.__n_nodes[i + 1]) :
                    c.append(random.uniform(-1, 1))
                    c_bef.append(0)
                b.append(c)
                b_bef.append(c_bef)
            self.__weights.append(b)
            self.__weights_bef.append(b_bef)

    def __sigmoid(self, v) :
        return 1/(1 + np.exp(-v))

    def __psi_apostrophe(self, value) :
        return value * (1 - value)

    def __generate_batch(self) :
        total_row = len(self.__X_train)

        index = [x for x in range(total_row)]
        self.__indexes = []

        random.shuffle(index)

        while len(index) >= self.__batch_size :
            temp = []
            for i in range (self.__batch_size) :
                temp.append(index.pop())
            self.__indexes.append(temp)
        self.__indexes.append(index)

    def __forward_pass(self, is_predict) :
        batch = self.__batch_X.values

        self.__outputs = [] # initialize output to zero

        for row_idx in range(len(self.__batch_X)) : # iterate for each row
            row = [] # for output in each row

            for layer_idx in range (self.__hidden_layer + 1) : # iterate for each layer
                layer = [] # for output in each layer

                for node_idx in range (self.__n_nodes[layer_idx + 1]) : # iterate for each node in output layer
                    node_v = 0

                    for input_idx in range(self.__n_nodes[layer_idx]) : # iterate for input node from input layer, +1 for 
                        input_value = 0
                        if input_idx == 0 : # bias
                            input_value = 1
                        else :
                            if layer_idx == 0 : # first layer (input from dataset)
                                input_value = batch[row_idx][input_idx]
                            else : # input from output before
                                input_value = row[layer_idx - 1][input_idx]
                        
                        node_v += input_value * self.__weights[layer_idx][input_idx][node_idx]

                    if layer_idx == self.__hidden_layer :
                        if is_predict :
                            node_v = self.__sigmoid(node_v)
                        else :
                            node_v = round(self.__sigmoid(node_v))
                    else :
                        # print (node_v)
                        node_v = self.__sigmoid(node_v)

                    layer.append(node_v)
                row.append(layer)
            self.__outputs.append(row)

    def __backward_pass(self) :
        # initialize errors to all zero
        self.__errors = []

        for idx, output in enumerate(self.__outputs) :
            temp_error = []
            # output layer
            o_idx = self.__hidden_layer
            o_predict = output[o_idx][0]
            # print(self.__psi_apostrophe(o_predict) * (self.__batch_y[idx] - o_predict))
            temp_error.insert(0, [self.__psi_apostrophe(o_predict) * (self.__batch_y[idx] - o_predict)]) # for error output layer

            # hidden layer
            # -2 dari -1 karena index dimulai dari 0
            #         -1 karena tidak pakai output dari layer output
            for i in range (len(output) -2, -1, -1) :

                # perkalian matriks
                matrix_error = np.matrix(temp_error[0])
                matrix_weight = np.matrix(self.__weights[i + 1])

                result = matrix_weight.dot(matrix_error.T)
                result = np.squeeze(np.asarray(result.T)).tolist()

                del result[0]

                temp_error.insert(0, list(map(lambda x, y: self.__psi_apostrophe(x) + y, output[i], result)))

            # append output
            self.__errors.append(temp_error)

    def __update_weights(self) :
        temp_weights = self.__weights

        # delta weight
        delta_weights = []
        for i in range(len(self.__weights)) : # iterate layer
            delta_weight = []
            for j in range(len(self.__weights[i])): # iterate node input
                deltas = []
                for k in range(len(self.__weights[i][j])) : # iterate node output
                    delta = 0 # variable to sum all delta weight
                    for idx in range (len(self.__outputs)) : # iterate row in a batch
                        self.__outputs[idx].insert(0, self.__batch_X.iloc[idx].tolist())
                        self.__outputs[idx][i].insert(0, 1)
                        delta += (self.__momentum * self.__weights_bef[i][j][k]) + (self.__learning_rate * self.__errors[idx][i][k] * self.__outputs[idx][i][j])
                        del self.__outputs[idx][0]
                    deltas.append(delta)
                delta_weight.append(deltas)
            delta_weights.append(delta_weight)

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

        if isinstance(y, pd.core.series.Series) :
            y = y.tolist()

        # raise error
        # if not isinstance(X, pd.core.frame.DataFrame) :
        #     raise TypeError("X must be a pandas.core.frame.DataFrame")
        # elif not all(isinstance(x, IntegerTypes) for x in y) :
        #     raise TypeError("y must be a list of integer")
        # elif X.select_dtypes(exclude=['number']).empty :
        #     raise TypeError("X must be all number")

        self.__X_train = X
        self.__y_train = y
        self.__random_weights()

        for _ in range (self.__epoch) :
            self.__generate_batch()
            for j in range (len(self.__indexes)) :
                self.__batch_X = X.iloc[self.__indexes[j], :]
                self.__batch_y = [self.__y_train[x] for x in self.__indexes[j]]
                self.__forward_pass(True)
                self.__backward_pass()
                self.__update_weights()

    def predict(self, X) :
        # X is pandas.dataframe

        # if not isinstance(X, pd.core.frame.DataFrame) :
        #     raise TypeError("X must be a pandas.core.frame.DataFrame")
        # elif X.select_dtypes(exclude=['number']).empty :
        #     raise TypeError("X must be all number")

        self.__batch_X = X
        self.__forward_pass(False)
        result = list(map(lambda x: round(x[self.__hidden_layer][0]), self.__outputs))
        return result