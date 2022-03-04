from sklearn.cluster import KMeans
import numpy as np


class UnsupportedModel(Exception):

    def __init__(self, estimator_name: str):
        self.message = f'Unsupported estimator {estimator_name}'
        super().__init__(self.message)


def get_supportded_estimator() -> dict:
    return {
        'kmeans': KMeans,
    }


def train(x: np.array, estimator_name: str):
    estimators = get_supportded_estimator()

    if estimator_name not in estimators:
        raise UnsupportedModel(estimator_name)

    estimator = estimators[estimator_name]()
    estimator.fit(x)

    return estimator
