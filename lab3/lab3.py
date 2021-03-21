import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        print(self.pima.head())
        self.X_test = None
        self.y_test = None


    def define_feature(self):
        """
        Defines the features to use from the supplied data set.
        :return: the feature matrix X and the label vector y
        """
        # Previously using just a few columns..
        #feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
        # Now using all available diabetes data as features
        feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
        X = self.pima[feature_cols]
        y = self.pima.label
        return X, y

    def train(self):
        """
        Trains the classification model. Applies a standard scaler to the feature matrix,
        splits the data, then trains the model using the training data.
        :return: the trained logistic regression model
        """
        # split X and y into training and testing sets
        X, y = self.define_feature()

        # scale the features before training the model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # partition the data set into training and test sets
        X_train, self.X_test, y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.33, shuffle=False, random_state=0)

        # train a logistic regression model on the training set
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        return logreg

    def predict(self):
        """
        Generates predictions using the test data set generated during training.
        :return: the predictions vector
        """
        model = self.train()
        y_pred_class = model.predict(self.X_test)
        return y_pred_class

    def calculate_accuracy(self, result):
        """
        Calculates the accuracy of the predictions as compared to the test label data.
        :param result: the predictions vector that will be evaluated
        :return: the accuracy score
        """
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        """
        Outputs some distribution info about the test data set.
        :return: the test data mean
        """
        dist = self.y_test.value_counts()
        print('Distribution:')
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        print(f"% of ones: {percent_of_ones}")
        print(f"% of zeros: {percent_of_zeros}")
        return self.y_test.mean()

    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)
    
if __name__ == "__main__":

    classifier = DiabetesClassifier()

    result = classifier.predict()
    print(f"Prediction={result}")

    score = classifier.calculate_accuracy(result)
    print(f"score={score}")

    con_matrix = classifier.confusion_matrix(result)
    print(f"confusion_matrix:\n{con_matrix}")

    y_test_mean = classifier.examine()
    print(f"y_test_mean={y_test_mean}")
