"""Anomaly detectors fitted on normal-class features.

All detectors share the same interface:
    detector.fit(train_features)            # (N, D)
    scores = detector.score(test_features)  # (M,)
Higher scores are more anomalous.
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors


class KNNDetector:
    """Mean distance to the k nearest normal training samples."""

    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric
        self._nn = None

    def fit(self, train_features):
        self._nn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        self._nn.fit(train_features)
        return self

    def score(self, test_features):
        if self._nn is None:
            raise RuntimeError("Call fit() before score().")
        distances, _ = self._nn.kneighbors(test_features)
        return distances.mean(axis=1)

    @property
    def name(self):
        return f"kNN(k={self.k})"


class MahalanobisDetector:
    """Mahalanobis distance from the normal-class Gaussian.

    Uses Ledoit-Wolf shrinkage to stabilise the covariance estimate when
    feature dimension is high relative to sample count.
    """

    def __init__(self):
        self._mean = None
        self._inv_cov = None

    def fit(self, train_features):
        self._mean = train_features.mean(axis=0)
        lw = LedoitWolf(assume_centered=False).fit(train_features)
        self._inv_cov = lw.precision_
        return self

    def score(self, test_features):
        if self._mean is None or self._inv_cov is None:
            raise RuntimeError("Call fit() before score().")
        diff = test_features - self._mean
        left = diff @ self._inv_cov
        return np.sqrt(np.einsum("nd,nd->n", left, diff))

    @property
    def name(self):
        return "Mahalanobis"


class KMeansDetector:
    """Distance to the nearest KMeans centroid of the normal training set."""

    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._kmeans = None

    def fit(self, train_features):
        k = min(self.n_clusters, len(train_features))
        self._kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
        self._kmeans.fit(train_features)
        return self

    def score(self, test_features):
        if self._kmeans is None:
            raise RuntimeError("Call fit() before score().")
        centers = self._kmeans.cluster_centers_
        return cdist(test_features, centers, metric="euclidean").min(axis=1)

    @property
    def name(self):
        return f"KMeans(k={self.n_clusters})"


def build_detectors(knn_k=5, kmeans_ks=(3, 5, 10)):
    """Return one of each detector type, with KMeans for each value of k."""
    detectors = [KNNDetector(k=knn_k), MahalanobisDetector()]
    for k in kmeans_ks:
        detectors.append(KMeansDetector(n_clusters=k))
    return detectors


def fit_and_score(train_features, test_features, knn_k=5, kmeans_ks=(3, 5, 10)):
    """Fit all detectors on train_features and return their scores on test_features."""
    results = {}
    for det in build_detectors(knn_k=knn_k, kmeans_ks=list(kmeans_ks)):
        det.fit(train_features)
        results[det.name] = det.score(test_features)
    return results
