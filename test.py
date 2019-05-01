from model import Model
from dataset import Dataset
from storage import Storage

# Set backend as hacky fix (virtualenv, matplotlib error)
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import sys
import numpy as np


def run_training(hidden, train_X, train_y):
    """
    ...

    Parameters
    ----------

    Returns
    -------
    """
    model = Model(
            input_size=2,
            output_size=1,
            hidden_layers=hidden,
            activation='tanh')

    history = model.train(train_X, train_y, epochs=TRAIN_EPOCHS, verbose=0)

    return model, history

def run_testing(model, test_X, test_y):
    """
    ...

    Parameters
    ----------

    Returns
    -------
    """
    scores = model.mlp.evaluate(test_X, test_y, verbose=0)

    test_loss = scores[0]
    test_acc = scores[1]

    return test_loss, test_acc

def add_results_to_store(store, length, hidden, all_test_loss, all_test_acc):
    """
    ...

    Parameters
    ----------

    Returns
    -------
    """
    avg_loss = np.mean(all_test_loss)
    std_loss = np.std(all_test_loss)
    min_loss = np.min(all_test_loss)
    max_loss = np.max(all_test_loss)
    avg_acc = np.mean(all_test_acc)
    std_acc = np.std(all_test_acc)
    min_acc = np.min(all_test_acc)
    max_acc = np.max(all_test_acc)

    results_dict = dict(zip(store.cols, [
            length,
            hidden,
            avg_loss,
            std_loss,
            min_loss,
            max_loss,
            avg_acc,
            std_acc,
            min_acc,
            max_acc
        ]))

    return store.add_result(results_dict)

TRAIN_NOISE = None
TRAIN_EPOCHS = None
NUMBER_OF_TESTS_PER_MODEL = None

if __name__ == '__main__':
    # Get global arguments from user input
    try:
        TRAIN_NOISE = float(sys.argv[1])
        TRAIN_EPOCHS = int(sys.argv[2])
        NUMBER_OF_TESTS_PER_MODEL = int(sys.argv[3])
    except:
        print ('python test.py [noise_value] [epochs] [num_test_per_model]')
        exit()

    # The tests to perform hard coded in
    # TODO: Maybe get tests to perform from csv file?
    tests = {
        'train_length' : [16, 32, 64],
        'hidden_neurons' : [(2,), (4,), (8,)]
    }

    # The results that will be stored per test
    cols =[
            'Training_Length',
            'Hidden_Neurons',
            'Avg_Test_Loss',
            'Std_Test_Loss',
            'Min_Test_Loss',
            'Max_Test_Loss',
            'Avg_Test_Acc',
            'Std_Test_Acc',
            'Min_Test_Acc',
            'Max_Test_Acc',
        ]

    # Initialise storage to hold the results
    store = Storage(cols)

    # --------------- BUILD OUTPUT GRAPHS ---------------------

    # Build the output graphs that will display all test results
    n_rows = len(tests['train_length'])
    n_cols = len(tests['hidden_neurons'])
    rows = ['# Training : {}'.format(i) for i in tests['train_length']]
    cols = ['# Hidden: {}'.format(i) for i in tests['hidden_neurons']]

    # Axes1 for the heatmaps, axes2 for the loss vs epoch line graphs
    # Axes 3 for loss boxplots, axes4 for accuracy boxplots
    fig1, axes1 = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    fig2, axes2 = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    fig3, axes3 = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    fig4, axes4 = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)

    fig1.suptitle("Heatmaps")
    fig2.suptitle("Convergence graphs")
    fig3.suptitle("Loss boxplots")
    fig4.suptitle("Accuracy boxplots")

    # Set labels for axes
    for i in range(n_rows):
        axes1[i][0].set_ylabel(rows[i])
        axes2[i][0].set_ylabel(rows[i])
        axes3[i][0].set_ylabel(rows[i])
        axes4[i][0].set_ylabel(rows[i])

    for i in range(n_cols):
        axes1[n_rows-1][i].set_xlabel(cols[i])
        axes2[n_rows-1][i].set_xlabel(cols[i])
        axes3[n_rows-1][i].set_xlabel(cols[i])
        axes4[n_rows-1][i].set_xlabel(cols[i])

    # --------------- GENERATE DATA TO USE ---------------------

    # Generate training and test data
    train_data = Dataset(noise=TRAIN_NOISE)
    test_data = Dataset(noise=0.)
    test_X, test_y = test_data.generate_vectors(64)

    # --------------- RUN ALL TESTS ---------------------
    test_num = 1

    for i, length in enumerate(tests['train_length']):
        for j, hidden in enumerate(tests['hidden_neurons']):

            # Store every test result (loss, acc) in a list
            all_test_loss = np.empty(NUMBER_OF_TESTS_PER_MODEL)
            all_test_acc = np.empty(NUMBER_OF_TESTS_PER_MODEL)

            # Run many tests for many models of same architecture
            # Done to see how well the architecture performs on average
            # Not on how a single model performs
            for x in range(NUMBER_OF_TESTS_PER_MODEL):
                print (test_num, x)

                train_X, train_y = train_data.generate_vectors(length=length)

                # Train and test the model
                model, history = run_training(hidden, train_X, train_y)
                test_loss, test_acc = run_testing(model, test_X, test_y)

                # Store single result
                all_test_acc[x] = test_acc
                all_test_loss[x] = test_loss

            # Store model architecture results
            add_results_to_store(store, length, hidden, all_test_loss, all_test_acc)

            # Add results to graphs
            # YES, it only shows the last model trained
            # It's hard to show 20 different models!
            # TODO: Could show all convergence graphs of all models on one graph?
            img = model.heatmap(dimension=20)
            axes1[i, j].imshow(img, cmap=plt.get_cmap('binary'))
            axes2[i, j].plot(history.history['loss'])
            axes3[i, j].boxplot(all_test_loss)
            axes4[i, j].boxplot(all_test_acc)

            test_num += 1

    store.save_results('results.csv')

    fig1.savefig('images/heatmaps_{}_{}_{}.png'.format(TRAIN_NOISE, TRAIN_EPOCHS, NUMBER_OF_TESTS_PER_MODEL))
    fig2.savefig('images/convergence_graphs_{}_{}_{}.png'.format(TRAIN_NOISE, TRAIN_EPOCHS, NUMBER_OF_TESTS_PER_MODEL))
    fig3.savefig('images/loss_boxplots_{}_{}_{}.png'.format(TRAIN_NOISE, TRAIN_EPOCHS, NUMBER_OF_TESTS_PER_MODEL))
    fig4.savefig('images/accuracy_boxplots_{}_{}_{}.png'.format(TRAIN_NOISE, TRAIN_EPOCHS, NUMBER_OF_TESTS_PER_MODEL))
