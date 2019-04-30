import numpy as np

import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras import optimizers

class Model:

    def __init__(self, input_size=2, output_size=1, hidden_layers=(2,), activation='sigmoid'):
        """
        Constructor for the model class

        Builds the model with the given attributes.

        Parameters
        ----------
        input_size : int
            Number of input values that the model requires

        output_size : int
            Number of output values that the model requires

        hidden_layers : tuple
            Tuple containing the hidden layers architecture.
            For example hidden_layers = (4,4) is two hidden layers,
            with each layer containing 4 neurons.

        activation : string
            The activation function for the model. Only use the keras
            activation functions.
        """

        # Build the model
        mlp = Sequential()

        # Add the hidden layers
        for i, n in enumerate(hidden_layers):
            if (i == 0):
                mlp.add(Dense(n, activation=activation, input_dim=input_size))
            else:
                mlp.add(Dense(n, activation=activation))

        mlp.add(Dense(output_size, activation='sigmoid'))

        self.mlp = mlp

    def train(self, X, Y, batch_size=None, epochs=1000, lr=0.1, verbose=1):
        """
        Train the model with the given data

        Parameters
        ----------
        X : list
            Contains the training data
        Y : List
            Contains the target output of the training data
        lr : float
            The learning rate to optimise the training of the model with
        epochs : int
            The number of epochs to run the training for
        verbose : int
            Print training log or not (0 or 1)

        Returns
        -------
        history
            The history of training the model (loss vs epoch)
        """

        # Compile the model using mean_squared_error and given attributes
        loss = 'mean_squared_error'
        op = optimizers.SGD(lr=lr)
        self.mlp.compile(loss=loss, metrics=['accuracy'], optimizer=op)

        # Fit the model to the training data
        history = self.mlp.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=verbose)

        return history

    def heatmap(self, dimension):
        """
        Calculate the heatmap of the model.

        Parameters
        ----------
        dimension : int
            How many tests to make between 0 - 1

        Returns
        -------
        image
            An iamge able to display the heatmap
        """
        xs = np.zeros((dimension*dimension, 2))
        ys = np.zeros((dimension*dimension, 1))

        index = 0
        for i in range(dimension):
            for j in range(dimension):
                # Define the inputs
                x1 = i / dimension
                x2 = j / dimension

                xs[index][0] = x1
                xs[index][1] = x2
                ys[index] = 0

                index += 1

        ys = self.mlp.predict_proba(xs)

        img = np.array([i[0] for i in ys])
        img = np.reshape(img, (dimension, dimension))

        return img
