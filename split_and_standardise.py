import numpy as np
import math

def standardise(dataset, mean=None, std=None):
    if mean is None and std is None:
        mean, std = np.mean(dataset, axis=0), np.std(
            dataset, axis=0
        )  # get mean and standard deviation of dataset
    standardized_dataset = (dataset - mean) / std
    return standardized_dataset, (mean, std)


def standardise_multiple(*datasets):
    mean, std = None, None
    for dataset in datasets:
        dataset, (mean, std) = standardise(dataset, mean, std)
        yield dataset


def train_validation_test_split(dataset, non_train: float = 0.4):
    """
    Split the dataset into train, validation and test sets. The `split` function that this function calls is
    a home-made version of sklearn's train_test_split function.
    
    :param dataset: The dataset to split
    :param non_train: The percentage of the dataset that will be used for validation and testing
    :type non_train: float
    :return: A tuple of tuples.
    """
    X, y = dataset
    X_train, X_test, y_train, y_test = split(X, y, split_ratio=0.6)

    X_validation, X_test, y_validation, y_test = split(X_test, y_test, split_ratio=0.5)
    
    return (X_train, y_train), (X_validation, y_validation), (X_test, y_test)


def split(*datasets, split_ratio):
    """
    It takes in a list of datasets and a split ratio, and returns a list of datasets, where each dataset
    is split into two datasets, according to the split ratio. It is essentially the same thing as sklearn's
    `train_test_split` function.
    
    :param split_ratio: the ratio of the first dataset to the second dataset
    """
    #the split ratio is what goes in first
    n_samples = len(datasets[0])
    print(type(n_samples))
    n_first = math.floor(n_samples*split_ratio)
    print(type(n_first))
    choice = np.random.choice(n_samples, n_first, replace=False)
    indexes = np.zeros(n_samples, dtype=bool)
    indexes[choice] = True
    rest = ~indexes

    for dataset in datasets:
        assert len(dataset) == n_samples
        splitted_dataset_first = dataset[indexes]
        splitted_dataset_second = dataset[rest]
        yield(splitted_dataset_first)
        yield(splitted_dataset_second)

    
