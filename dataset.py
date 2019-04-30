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

        Vectors in form (v1, v2, t1) where v1, v2 are the two inputs, and t1 is the target output.
        Noise of the data determined by the noise value set for this class.

        Parameters
        ----------
        length : int
            The number of vectors in training data (default is 16)

        Returns
        -------
        list
            A list of vectors containing the data
        list
            A list of labels (0 or 1) containing the target result
        """

        def get_label(px, py):
            if ((px > 0.5 and py > 0.5) or (px < 0.5 and py < 0.5)):
                return 0
            else:
                return 1

        points = np.zeros([length, 2])
        labels = np.zeros(length).astype(int)

        for i in range(length):
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            noise_x = np.random.normal(0, self.noise)
            noise_y = np.random.normal(0, self.noise)
            labels[i] = get_label(x + noise_x, y + noise_y)
            points[i] = (x, y)

        return points, labels


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
