import numpy as np

class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered as a core point.
    """
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.core_sample_indices_ = None

    def fit(self, X):
        """
        Perform DBSCAN clustering from features or distance matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        for idx in range(n_samples):
            if visited[idx]:
                continue
            visited[idx] = True
            neighbors = self._region_query(X, idx)

            # If not a core point, mark as noise temporarily
            if len(neighbors) < self.min_samples:
                labels[idx] = -1
            else:
                # Expand cluster
                self._expand_cluster(X, labels, idx, neighbors, cluster_id, visited)
                cluster_id += 1

        # Record final labels and core samples
        self.labels_ = labels
        self.core_sample_indices_ = [
            i for i in range(n_samples)
            if len(self._region_query(X, i)) >= self.min_samples
        ]
        return self

    def fit_predict(self, X):
        """
        Convenience method; equivalent to calling fit(X) followed by returning labels_.
        """
        self.fit(X)
        return self.labels_

    def _region_query(self, X, idx):
        """
        Return indices of all points within eps of point idx.
        """
        distances = np.linalg.norm(X - X[idx], axis=1)
        return list(np.where(distances <= self.eps)[0])

    def _expand_cluster(self, X, labels, idx, neighbors, cluster_id, visited):
        """
        Grow the cluster with id `cluster_id` by adding all density-reachable points.
        """
        labels[idx] = cluster_id
        i = 0
        while i < len(neighbors):
            nb_idx = neighbors[i]
            if not visited[nb_idx]:
                visited[nb_idx] = True
                nb_neighbors = self._region_query(X, nb_idx)
                if len(nb_neighbors) >= self.min_samples:
                    # Add new neighbors to the list
                    for n in nb_neighbors:
                        if n not in neighbors:
                            neighbors.append(n)
            if labels[nb_idx] == -1:
                # Change noise to border point
                labels[nb_idx] = cluster_id
            i += 1