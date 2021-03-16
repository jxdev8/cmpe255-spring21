import numpy as np
from sklearn.linear_model import SGDClassifier
from os import path
import pandas as pd

np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

def plot_digit(some_digit):
    """
    Plots the supplied digit.
    """
    # reshape as a 28x28 image, each feature being a single pixel's intensity (0-255)
    some_digit_image = some_digit.reshape(28, 28)

    plt.imshow(some_digit_image, cmap='binary')
    plt.axis('off')
    plt.show()

def load_data():
    """
    Extracts and loads the MNIST data. The first time it pulls the data, it saves it
    into local files (X.csv and y.csv). Subsequent calls will load the local csv files
    to avoid downloading them again.
    :return: The feature matrix X, and the target labels y
    """
    if path.exists('X.csv') and path.exists('y.csv'):
        print('reading from local csv files..')
        X = pd.read_csv('X.csv', header=None)
        y = pd.read_csv('y.csv', header=None)
        y = y.values.ravel()
    else:
        try:
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml('mnist_784', version=1, cache=True)
            mnist.target = mnist.target.astype(np.uint8) # fetch_openml() returns targets as strings
            # Note: Commented out, no need to sort, we want the randomized set for SGD
            # sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
        except ImportError:
            from sklearn.datasets import fetch_mldata
            mnist = fetch_mldata('MNIST original')
        X, y = mnist['data'], mnist['target']
        np.savetxt(fname='X.csv', X=X, delimiter=',', fmt='%d')
        np.savetxt(fname='y.csv', X=y, delimiter=',', fmt='%d')
    return X, y

def train_binary_classifier_model(X_train, y_train, digit_to_detect):
    """
    Trains a Stochastic Gradient Descent (SGD) classifier, which is good at handling
    very large data sets, and deals with training instances independently.
    Note: SGD classifier relies on randomness, hence why it's termed Stochastic.
    :param X_train: The feature matrix to use for training the model
    :param y_train: The labels vector to use for training the model
    :param digit_to_detect: The specific digit to train the binary classifier on
    :return:
    """
    y_train_digit = (y_train == digit_to_detect)
    sgd = SGDClassifier(random_state=42)
    sgd.fit(X_train, y_train_digit)
    return sgd, y_train_digit

def calculate_cross_val_score(sgd, X, y):
    """
    Runs a k-fold (k=3) cross-validation on the supplied data.
    :param sgd: The Stochastic Gradient Classifier model to use for evaluation.
    :param X: The features matrix
    :param y: The labels vector
    """
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(sgd, X, y, cv=3, scoring='accuracy')
    print('cross-val scores: ', scores)


if __name__ == "__main__":

    # fetch the MNIST data into feature matrix X and label data y
    # consists of 70000 samples
    X, y = load_data()

    # split the data into training (60k) and test (10k) data sets
    # note: MNIST data is already shuffled and ready to partition for training/test
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    # generate a SGD classifier based on the supplied training data
    sgd, y_train_digit = train_binary_classifier_model(X_train, y_train, 5)

    # test out a prediction from the trained SGD model
    some_digit = X.iloc[0]
    pred = sgd.predict([some_digit])
    print('single digit prediction: ', pred, ', actual digit: ', y_train[0])
    plot_digit(np.array(some_digit))

    # get predictions from the test data set
    y_preds = sgd.predict(X_test)
    n_correct = sum(y_preds == (y_test == 5))
    print("accuracy of test data predictions: ", n_correct / len(y_preds))

    # evaluate accuracy using k-fold cross-validation with k=3
    calculate_cross_val_score(sgd, X, y)
