from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


class PFA(object):
    def __init__(self, n_features, q=None):
        self.q = q
        self.n_features = n_features

    def fit(self, X):
        if not self.q:
            self.q = X.shape[0]

        sc = StandardScaler()
        X = sc.fit_transform(X)
        pca = PCA(n_components=self.q).fit(X)
        A_q = pca.components_.T

        kmeans = KMeans(n_clusters=self.n_features).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]


X = pd.read_csv("../features_analysis_all_filtered.csv")
X = X.to_numpy()
pfa = PFA(n_features=20)
pfa.fit(X)

# To get the transformed matrix
X = pfa.features_
# print(X)
# To get the column indices of the kept features
column_indices = pfa.indices_

X = pd.read_csv("../features_analysis_all_filtered.csv")
features = []
for idx in column_indices:
    features.append(X.columns[idx])
print(column_indices)
# print(features)
