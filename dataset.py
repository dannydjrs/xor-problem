import numpy as np

# HACKY FIX
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import sys

class Dataset:

    def __init__(self, noise=0.25):
        """
        Constructor for the dataset class

        Parameters
        ----------
        noise : float
            Variance for the noise of the data (default is 1/4)
        """
        self.noise = noise


    def generate_vectors(self, length=16):
        """
        Generates the data for the xor as array of vectors.
        Same number of each target in output vectors.

        Vectors in form (v1, v2, t1) where v1, v2 are the two inputs, and t1 is the target output.
        Noise of the data determined by the noise value set for this class.

        Parameters
        ----------
        length : int
            The number of vectors in training data (default is 16)
            Length must be divisible by 4 for it to return the given number

        Returns
        -------
        list
            A list of vectors containing the data
        list
            A list of labels (0 or 1) containing the target result
        """
        # Base x and y for xor
        init_x = [[0, 0],[0, 1],[1, 0],[1, 1]]
        init_y = [0, 1, 1, 0]

        # Generate same number of each vector with added noise
        x = np.repeat(init_x, [length//4], axis=0)
        y = np.repeat(init_y, [length//4], axis=0)
        noise = np.random.normal(loc=0, scale=self.noise, size=(len(x), 2))
        points = x + noise

        return points, y


def test(noise=0.25, length=16):
    data = Dataset(noise=noise)

    train_X, train_y = data.generate_vectors(length=length)

    plot_points(train_X, train_y)

def plot_points(points, labels):
    plt.scatter([x for [x,y] in points], [y for [x,y] in points], c=['red' if t == 0 else 'blue' for t in labels] )
    plt.show()


if __name__ == '__main__':

    noise = None
    length = None

    try:
        noise = float(sys.argv[1])
        length = int(sys.argv[2])
    except:
        print ('python dataset.py [noise_value] [number_of_vectors]')
        exit()

    test(noise, length)
