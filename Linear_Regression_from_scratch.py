import numpy as np
from aicore.ml import data
from split_and_standardise import *

import matplotlib.pyplot as plt
def plot_loss(losses):
    """Helper function for plotting the loss against the epoch"""
    plt.figure() # make a figure
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.plot(losses) # plot costs
    plt.show()

def mse_loss(y_hat, labels): # define the criterion (loss function)
    errors = y_hat - labels ## calculate the errors
    squared_errors = errors ** 2 ## square the errors
    mean_squared_error = sum(squared_errors) / len(squared_errors) ## calculate the mean 
    return mean_squared_error # return the loss

def calculate_loss(model, X, y):
    return mse_loss(model.predict(X), y)


class myGDLinearRegression:

    def init(self):

        self.coef_ = None
        self.intercept_ = None

    def gradient_descent_step(self, features, predictions, true_labels, learning_rate):

        size = len(predictions)
        difference = predictions - true_labels # Calculating the errors
        loss_deriv_wrt_weights = 2 * np.sum(features.T * difference) / size
        loss_deriv_wrt_bias = 2 * np.sum(difference) / size

        new_weights = self.coef_ - learning_rate * loss_deriv_wrt_weights
        new_bias = self.intercept_ - learning_rate * loss_deriv_wrt_bias

        return new_weights, new_bias       
    
    
    def fit(self, X_train, y_train, n_epochs=500, learning_rate=0.001/32, batch_size=32, non_stochastic=False):
        all_costs =[]
        n_features = X_train.shape[1]  # The number of features is the number of columns in our data set.

        try:
            n_pred_targets = y_train.shape[1] # If y is not one-dimensional then we assume the number of columns corresponds to the number of prediction targets (k),
        except:                               # as opposed to the number of data entries.
            n_pred_targets = 1


        self.coef_ = np.random.randn(n_features, n_pred_targets) # W
        self.intercept_ = np.random.randn(n_pred_targets) # b
        
        for _ in range(n_epochs):
            print(_)
            for batch in self.iterate_minibatches(X_train, y_train, batch_size, non_stochastic, shuffle=False):
                X_btrain, y_btrain = batch
                b_predics = self.predict(X_btrain)
                new_weight, new_bias = self.gradient_descent_step(X_btrain, b_predics, y_btrain, learning_rate)
                self.update_parameters(new_weight, new_bias)
            predic_snapshot = self.predict(X_train)
            all_costs.append(mse_loss(predic_snapshot, y_train))

        plot_loss(all_costs)
        print('Final loss:', all_costs[-1])
        print('Weight values:', self.coef_)
        print('Bias values:', self.intercept_)
        return self

    @staticmethod
    def iterate_minibatches(X_train, y_train, batch_size, non_stochastic, shuffle=True):  # Modified version of something I found on stackexchange. Its a neat way of getting all the minibatches.
        """
        It takes in the training data, the training labels, the batch size, and a boolean value for
        whether or not you want to use stochastic gradient descent. If you want to use stochastic
        gradient descent, it will return a random batch of the training data and labels. If you don't
        want to use stochastic gradient descent, it will return the entire training data and labels.
        
        :param X_train: The training data
        :param y_train: The training labels
        :param batch_size: The number of samples to use in each minibatch
        :param non_stochastic: If True, then the entire dataset is used for each epoch. If False, then
        the dataset is split into minibatches
        :param shuffle: Whether to shuffle the data before each epoch, defaults to True (optional)
        """
        
        assert X_train.shape[0] == y_train.shape[0]
        if non_stochastic:
            yield X_train, y_train
        else:

            if shuffle:
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
            for batch_start in range(0, X_train.shape[0], batch_size):
                batch_end = min(batch_start + batch_size, X_train.shape[0])
                if shuffle:
                    excerpt = indices[batch_start:batch_end]
                else:
                    excerpt = slice(batch_start, batch_end)
                yield X_train[excerpt], y_train[excerpt]

    def update_parameters(self, weights, bias):
        """
        > The function takes in the weights and bias and updates the parameters of the model.
        
        :param weights: the weights of the model
        :param bias: the bias term
        """
        self.coef_ = weights
        self.intercept_ = bias

    def predict(self, X):
        """
        It predicts the value of y for a given value of x, based on the current parameters of the model.
        
        :param X: the input data
        :return: The sum of the predicted values for each row of X.
        """
        y_pred = X @ self.coef_ + self.intercept_
        return y_pred.sum(axis=1)



if __name__ == "__main__":
    '''Here we show an implementation of the model using one of sklearn's inbuilt datasets. This is the only time sklearn is used.
    Note that the perforamnce of the model is definitely not as good as sklearn's inbuilt LinearRegression model. This is because the 
    use of the gradient descent method is causing the loss surface to converge to a minimum which is not the global minimum.'''

    from sklearn import datasets
    np.random.seed(4)
    # Use `train_validation_test_split` to split the data into training, validation, and test sets.
    (X_train, y_train), (X_validation, y_validation), (X_test, y_test) = train_validation_test_split(datasets.fetch_california_housing(return_X_y=True)) #default seems to be to allocate 60% to training, 20% to each of test and validation
    X_train, X_validation, X_test = standardise_multiple(X_train, X_validation, X_test)
    model = myGDLinearRegression()
    model.fit(X_train, y_train)

    print(f"Validation loss after training: {calculate_loss(model, X_validation, y_validation)}")
    print(f"Test loss after training: {calculate_loss(model, X_test, y_test)}")

