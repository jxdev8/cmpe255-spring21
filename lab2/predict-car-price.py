import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

class CarPrice:

    def __init__(self):

        # ingest the input data and do some naming cleanup
        self.df = pd.read_csv('data/data.csv')
        print(f'{len(self.df)} lines loaded')
        self.trim()

        # features we want to use for predictions
        self.base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']

        # set display options
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def display(self, title, X, y, y_pred):
        print('========', title, '========')
        columns = ['engine_cylinders','transmission_type','driven_wheels','number_of_doors',
                   'market_category','vehicle_size','vehicle_style','highway_mpg','city_mpg','popularity']
        X = X.copy()
        X = X[columns]
        X['msrp'] = y.round(2)
        X['msrp_pred'] = y_pred.round(2)
        print(X.head(5).to_string(index=False))

    def get_data_subsets(self):
        # divide up the total entries into: 20% validation, 20% test, and 60% training
        n = len(self.df)
        n_val = int(n * 0.2)
        n_test = int(n * 0.2)
        n_train = n - (n_val + n_test)

        # shuffle the entries based on randomizing idx array values
        idx = np.arange(n)
        np.random.shuffle(idx)
        df_shuffled = self.df.iloc[idx]

        # slice up the shuffled dataframe into training, validation, and test subsets
        return [df_shuffled.iloc[:n_train].copy(),
                df_shuffled.iloc[n_train:n_train+n_val].copy(),
                df_shuffled.iloc[n_train+n_val:].copy()]

    def get_label_data(self, df):
        """
        Extracts and normalizes the label data from the supplied dataframe.

        :param df: The input dataframe to pull data from
        :return: The original label vector along with its normalized label vector
        """
        y_orig = df.msrp.values
        y = np.log1p(df.msrp.values)
        del df['msrp']
        return y_orig, y

    def prepare_X(self, df):
        """
        Extracts the feature data from the supplied dataframe based on the features desired.

        :param df: The input dataframe to pull data from
        :return: The data representing the input feature matrix to use
        """
        df_num = df[self.base]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X

    def validate(self, y, y_pred):
        """
        Calculates the root mean-squared error between the label data and prediction data

        :param y: Label data vector
        :param y_pred: Predicted data vector
        :return: The root mean-squared scalar value
        """
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)

    def plot_actual_vs_prediction(self, header, y, y_pred):
        """
        Plot the label data against the predicted data

        :param header: Additional string to display for the title
        :param y: Label data vector
        :param y_pred: Predicted data vector
        """
        sns.distplot(y, label='target', kde=False,
                     hist_kws=dict(color='red', alpha=0.6))
        sns.distplot(y_pred, label='prediction', kde=False,
                     hist_kws=dict(color='blue', alpha=0.8))
        plt.legend()
        plt.ylabel('Frequency')
        plt.xlabel('Log(Price + 1)')
        plt.title('{0}Predictions vs Actual Distribution'.format(header))
        plt.show()

    def linear_regression(self, X, y):
        """
        Use the Normal Equation to get the optimal parameter vector w
        Normal Equation = inverse(Xtranspose . X) . Xtranspose . y

        :param X: feature matrix
        :param y: labels vector
        :return: optimal parameters (weights) based on the Normal Equation
        """
        ones = np.ones(X.shape[0])      # create a nx1 vector of ones
        X = np.column_stack([ones, X])  # adjust X to an m x (n+1) matrix, m=#samples, n=#features, for bias terms

        # Use the Normal Equation to get the optimal parameter vector w
        # Normal Equation = inverse(Xtranspose . X) . Xtranspose . y
        XTX = X.T.dot(X)              # (Xtranspose) . X
        XTX_inv = np.linalg.inv(XTX)  # inverse of XTX
        w = XTX_inv.dot(X.T).dot(y)

        return w[0], w[1:]

if __name__ == "__main__":
    # execute only if run as a script
    cp = CarPrice()

    # divide up the data into 3 subsets, for training, validation, testing
    df_train, df_val, df_test = cp.get_data_subsets()
    print('got subsets of data: ', len(df_train), len(df_val), len(df_test))

    # get the label data for each subset of samples
    y_train_orig, y_train = cp.get_label_data(df_train)
    y_val_orig, y_val = cp.get_label_data(df_val)
    y_test_orig, y_test = cp.get_label_data(df_test)

    # get the features to use for each subset of data
    X_train = cp.prepare_X(df_train)
    X_val = cp.prepare_X(df_val)
    X_test = cp.prepare_X(df_test)

    # use training data to generate parameter (weights) vector
    w_0, w = cp.linear_regression(X_train, y_train)
    # generate prediction vector based on features and weights
    y_train_pred = w_0 + X_train.dot(w)

    # generate predictions for validation data set
    y_val_pred = w_0 + X_val.dot(w)

    # generate predictions for test data set
    y_test_pred = w_0 + X_test.dot(w)

    # output 5 rows for each data subset based on the model
    cp.display('Training', df_train, y_train, y_train_pred)
    cp.display('Validation', df_val, y_val, y_val_pred)
    cp.display('Test', df_test, y_test, y_test_pred)

    # Display prediction performance for each phase
    perf_train = cp.validate(y_train, y_train_pred)
    perf_val = cp.validate(y_val, y_val_pred)
    perf_test = cp.validate(y_test, y_test_pred)
    print('\nTraining rmse: ', round(perf_train,4))
    print('Validation rmse: ', round(perf_val,4))
    print('Test rmse: ', round(perf_test,4))

    # plot the performance for each data subset based on the model
    cp.plot_actual_vs_prediction('Training: ', y_train, y_train_pred)
    cp.plot_actual_vs_prediction('Validation: ', y_val, y_val_pred)
    cp.plot_actual_vs_prediction('Test: ', y_test, y_test_pred)
