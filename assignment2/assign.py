#
# CMPE-255 / Spring 2021
# Assignment #2
# Date: 05/02/2021
# Name: John Monsod
# Student ID# 015234505
#

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class FaceRecognition:

    def __init__(self):
        self.faces = None
        self.num_samples = -1
        self.X = None
        self.y = None
        self.num_features = -1
        self.target_names = []
        self.num_classes = []

    def load_data(self):
        """
        Loads the face training/test data
        """
        self.faces = fetch_lfw_people(min_faces_per_person=60)
        print(f'Data has been loaded successfully, shape: {self.faces.images.shape}')

        self.num_samples, self.h, self.w = self.faces.images.shape
        self.X = self.faces.data
        self.y = self.faces.target
        self.num_features = self.X.shape[1]
        self.target_names = self.faces.target_names
        self.num_classes = self.target_names.shape[0]
        print(f'num_samples (images): {self.num_samples}')
        print(f'image size: {self.h} x {self.w}')
        print(f'num_features: {self.num_features}')
        print(f'target_names: {self.target_names}')

        return self.faces

    def split_data(self, test_size):
        """
        Splits the data based on the directed split size.
        :param test_size: The test data size relative to the whole data set.
        :return: The corresponding X and y training and test data sets after the split.
        """
        return train_test_split(self.X, self.y, test_size=test_size, random_state=42)

    def create_model_pipeline(self):
        """
        Creates the model pipeline based on a sequence of PCA then SVC.
        :return: The model pipeline that can be used for model training.
        """
        pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
        svc = SVC(kernel='rbf', class_weight='balanced')
        return make_pipeline(pca, svc)

    def train_gridsearch_model(self, pipe, X_train, y_train):
        """
        Trains GridSearch using the supplied estimators within the pipe, using the supplied training data.
        :param pipe: The model pipeline consisting of PCA and SVC to run the model into.
        :param X_train: The training data set.
        :param y_train: The targets for the training data set.
        :return: The trained model based on Grid Search determination of best hyperparams.
        """
        param_grid = {
            'svc__C': [1, 5, 10, 50, 1e3, 5e3, 1e4, 5e4, 1e5],
            'svc__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        }
        clf = GridSearchCV(pipe, param_grid)
        clf.fit(X_train, y_train)
        print('\nGridSearch done, best estimator:')
        print(clf.best_estimator_)
        return clf

    def plot_gallery(self, images, pred_names, truths, h, w, n_row, n_col):
        """
        Plots the supplied images with their associated predicted and actual names.
        :param images: The array of image data to display.
        :param pred_names: The associated predicted names for the images to display.
        :param truths: The associated actual names for the images to display.
        :param h: The height of the pixel count for a given image
        :param w: The width of the pixel count for a given image
        :param n_row: The number of rows for the subplots in the figure to display.
        :param n_col: The number of columns for the subplots in the figure to display.
        """
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            title = plt.title(pred_names[i], size=12)
            if truths[i]:
                plt.setp(title, color='black')
            else:
                plt.setp(title, color='red')
            plt.xticks(())
            plt.yticks(())
        plt.show()

    def get_name(self, idx):
        """
        Obtains the display name corresponding to the index position specified.
        :param idx: The index into the names to determine the name.
        :return: The name corresponding the index specified.
        """
        return self.target_names[idx]

    def display_classification_report(self, y_test, y_pred):
        """
        Displays the classification report containing the precision, recall, F1, and support data.
        """
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred, target_names=self.target_names))

    def display_confusion_matrix(self, y_test, y_pred):
        """
        Displays the confusion matrix heatmap.
        :param y_test: The actual target data from the test data set.
        :param y_pred: The predicted target data from running the model on the test data set.
        """
        conf_matrix = confusion_matrix(y_test, y_pred, labels=range(self.num_classes))
        conf_matrix_disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=self.target_names)
        conf_matrix_disp.plot(xticks_rotation='vertical')
        plt.show()

    def display_test_images_performance(self, X_test, y_test, y_pred):
        """
        Displays the images from the test data set, along with the name associated with each,
        colored red when the prediction is not equal to the actual, otherwise in black.
        :param X_test: The images from the test data set.
        :param y_test: The actual target data from the test data set.
        :param y_pred: The predicted target data from running the model on the test data set.
        """
        pred_names = [self.get_name(val) for val in y_pred]
        num_cols = 5
        num_rows = int(len(X_test) / num_cols)
        facerec.plot_gallery(X_test, pred_names, y_pred == y_test, self.h, self.w, num_rows, num_cols)


if __name__ == "__main__":

    facerec = FaceRecognition()

    # load the data needed
    faces = facerec.load_data()

    # split the data
    X_train, X_test, y_train, y_test = facerec.split_data(test_size=0.2)
    print(f'\nData has been split: training data={len(X_train)}, test data={len(X_test)}')

    # setup the steps to train the model, put into a pipeline
    pipe = facerec.create_model_pipeline()

    # use GridSearch to find the optimal hyperparams for modeling
    clf = facerec.train_gridsearch_model(pipe, X_train, y_train)

    # get predictions from the model trained
    y_pred = clf.predict(X_test)

    # display performance details
    facerec.display_classification_report(y_test, y_pred)

    # Confusion Matrix
    facerec.display_confusion_matrix(y_test, y_pred)

    # display performance of test images, with name in black for correctly predicted, red otherwise
    facerec.display_test_images_performance(X_test, y_test, y_pred)
