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

    def fit(self, X, y) :
        # X is pandas.dataframe
        # y is pandas.series

        return

    def predict(self, X) :
        # X is pandas.dataframe
        
        return