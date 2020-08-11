import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.exceptions import ConvergenceWarning
import warnings


class Context2DM(object):

    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab.itos)
        self.dm = None
        self.dm_dim = None

        self.corpus_reps_list = []
        self.word_reps_indices = [[] for _i in range(self.vocab_size)]

    def update(self, word, vector):
        if not np.isnan(vector).any():
            dm = np.outer(vector, vector)
            # normalise?
            self.dm[self.vocab.stoi[word]] += dm

    def add_word_rep(self, word_index, vector):
        # Add word representation from corpus, keep track of word indices in representation matrix
        if not np.isnan(vector).any():
            self.corpus_reps_list.append(vector)
            self.word_reps_indices[word_index].append(len(self.corpus_reps_list)-1)

    def cluster_word_reps(self, cluster_eval="sc", cluster_alg="kmeans"):
        # Cluster each word's corpus reps and replace with centroids
        cluster_centroids = []
        word_indices = [[] for _i in range(self.vocab_size)]

        score_func = silhouette_score if cluster_eval == "sc" else calinski_harabasz_score
        ClusterAlg = KMeans if cluster_alg == "kmeans" else AgglomerativeClustering

        for i, word in enumerate(self.vocab.itos):
            if i % 10000 == 0:
                print(i, len(self.vocab.itos))
            if len(self.word_reps_indices[i]) == 0:
                continue

            if len(self.word_reps_indices[i]) == 1:
                # Only one representation, so don't cluster
                cluster_centroids.append(self.corpus_reps_list[self.word_reps_indices[i][0]])
                word_indices[i].append(len(cluster_centroids) - 1)

            else:
                X = np.stack([self.corpus_reps_list[index] for index in self.word_reps_indices[i]])
                centroids = self.auto_cluster(X, ClusterAlg, score_func)
                for centroid in centroids:
                    cluster_centroids.append(centroid)
                    word_indices[i].append(len(cluster_centroids) - 1)

        self.corpus_reps_list = cluster_centroids
        self.word_reps_indices = word_indices

    def auto_cluster(self, X, ClusterAlg, score_func):

        max_score = -1
        max_model = None

        if np.isnan(X).any():
            print(len(X))
            for i in range(len(X)):
                print(X[i])

        for k in range(2, min(len(X)-1, 10) + 1):
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=ConvergenceWarning)
                k_model = ClusterAlg(n_clusters=k).fit(X)
            if len(set(k_model.labels_)) < k:
                # number of dictinct vectors is less than k
                break
            score = score_func(X, k_model.labels_)
            if score > max_score:
                max_score = score
                max_model = k_model

        if max_model is None:
            max_centroids = [np.mean(X, axis=0)]
        else:
            max_centroids = self.get_centroids(max_model, X)

        return max_centroids

    def get_centroids(self, model, X):
        if hasattr(model, "cluster_centers_"):
            return model.cluster_centers_
        else:
            centers = []
            for cluster in range(model.n_clusters_):
                cluster_vectors = np.stack([X[i] for i, label in enumerate(model.labels_) if label == cluster])
                cluster_center = np.mean(cluster_vectors, axis=0)
                centers.append(cluster_center)
            return centers

    def initialise_dms(self, dim=300):
        self.dm_dim = dim
        self.dm = np.zeros(shape=(self.vocab_size, self.dm_dim, self.dm_dim))

    def build_dms(self):
        # Compute word density matrices from corpus representations
        self.corpus_reps_list = np.stack(self.corpus_reps_list, axis=0)
        self.dm = np.zeros(shape=(self.vocab_size, self.dm_dim, self.dm_dim))
        for word_index in range(self.vocab_size):
            self.dm[word_index] = self.compute_dm(word_index)
        del self.corpus_reps_list # release from memory, since not needed again

    def compute_dm(self, word_index):
        reps = self.corpus_reps_list[self.word_reps_indices[word_index]]
        outer_products = np.einsum('ij,ik->ijk', reps, reps)
        dm = np.sum(outer_products, axis=0)
        return dm

    def normalise(self):
        # Normalise to unit trace
        self.dm = self.dm / np.reshape(self.dm.trace(axis1=1, axis2=2), (self.dm.shape[0], 1, 1))

    def contains(self, word):
        return word in self.vocab.stoi

    def get_dm(self, word):
        word_index = self.vocab.stoi[word]
        return self.dm[word_index]


if __name__=="__main__":
    A = np.array([[3, 1], [1, 3]])
    B = np.array([[49, 2], [12, 94]])
    print(Context2DM.fuzz(A, B))
