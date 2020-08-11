import torch
import pickle
import numpy as np
import scipy as sp
import math
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.exceptions import ConvergenceWarning
from nltk.corpus import stopwords
import warnings

# Define global constants
STOP_WORDS = stopwords.words("english")
STOP_WORDS.remove("few") # remove few since it's in the evaluation data
MAX_INPUT_LEN = 512

class Bert2DM(object):

    def __init__(self, vocab, corpus_reps=True):
        self.vocab = vocab
        self.vocab_size = len(vocab.itos)
        self.dm = None
        self.dm_dim = None

        if corpus_reps:
            self.corpus_reps_list = []
            self.word_reps_indices = [[] for _i in range(self.vocab_size)]
            self.corpus_reps = None
        else:
            self.initialise_dms(dim=768)

    """This is for not reducing DMs"""
    def update(self, word, vector):
        dm = np.outer(vector, vector)
        # normalise?
        self.dm[self.vocab.stoi[word]] += dm

    def initialise_dms(self, dim=300):
        self.dm_dim = dim
        self.dm = np.zeros(shape=(self.vocab_size, self.dm_dim, self.dm_dim))
    """Up to here"""

    def add_word_rep(self, word_index, vector):
        # Add word representation from corpus, keep track of word indices in representation matrix
        self.corpus_reps_list.append(vector)
        self.word_reps_indices[word_index].append(len(self.corpus_reps_list)-1)

    def reduce_word_reps(self, dim=30, reduce_alg="pca"):
        # Reduce corpus word representations with PCA
        self.dm_dim = dim
        X = np.stack(self.corpus_reps_list, axis=0)

        if reduce_alg == "pca":
            reducer = PCA(n_components=dim)
        elif reduce_alg == "svd":
            reducer = TruncatedSVD(n_components=dim)
        elif reduce_alg == "tsne":
            reducer = TSNE(n_components=dim, method="exact")
        self.corpus_reps = reducer.fit_transform(X)
        del self.corpus_reps_list # release from memory, since not needed again

    def cluster_word_reps(self, cluster_eval="sc", cluster_alg="kmeans"):
        # Cluster each word's corpus reps and replace with centroids
        cluster_centroids = []
        word_indices = [[] for _i in range(self.vocab_size)]

        score_func = silhouette_score if cluster_eval == "sc" else calinski_harabasz_score
        ClusterAlg = KMeans if cluster_alg == "kmeans" else AgglomerativeClustering

        for i, word in enumerate(self.vocab.itos):
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


    def build_dms(self):
        # Compute word density matrices from corpus representations
        self.dm = np.zeros(shape=(self.vocab_size, self.dm_dim, self.dm_dim))
        for word_index in range(self.vocab_size):
            self.dm[word_index] = self.compute_dm(word_index)
        del self.corpus_reps # release from memory, since not needed again

    def compute_dm(self, word_index):
        reps = self.corpus_reps[self.word_reps_indices[word_index]]
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

    def bert_process_sentence(self, tokens, ids, bert, wp_tokenizer, args):

        # Tokenize each word into word pieces and store to how many word pieces each word maps
        token_to_wp = []
        for vocab_token in tokens:
            wps = wp_tokenizer.tokenize(vocab_token)
            token_to_wp.append(len(wps))

        # Tokenize and encode sentence for BERT
        sentence = " ".join(tokens)
        input_ids = torch.tensor([wp_tokenizer.encode(sentence, add_special_tokens=True)])
        if len(input_ids[0]) > MAX_INPUT_LEN:
            return

        # Process sentence with BERT and extract word representations
        last_hidden_states = bert(input_ids)[2][12 - (args.extract_layers - 1):]
        if args.combine_layers == "no" or len(last_hidden_states) == 1:
            last_hidden_states = last_hidden_states[0]
        elif args.combine_layers == "concat":
            last_hidden_states = torch.cat(last_hidden_states, dim=2)
        elif args.combine_layers == "sum":
            last_hidden_states = torch.stack(last_hidden_states, dim=0)
            last_hidden_states = torch.sum(last_hidden_states, dim=0)

        # Go through sentence and store BERT representations, averaging word piece representations for a split-up word
        current_wp_index = 1  # offset for [CLS] token
        for j, word_index in enumerate(ids):
            if self.vocab.itos[word_index].isalpha() and (not args.remove_stopwords or self.vocab.itos[word_index] not in STOP_WORDS):
                word_reps = last_hidden_states[0, current_wp_index: current_wp_index + token_to_wp[j], :]
                word_rep = word_reps[0] if token_to_wp[j] == 1 else torch.mean(word_reps, dim=0)
                if args.reduce_dim:
                    self.add_word_rep(word_index, word_rep)
                else:
                    self.update(self.vocab.itos[word_index], word_rep)
            current_wp_index += token_to_wp[j]


if __name__=="__main__":
    A = np.array([[3, 1], [1, 3]])
    B = np.array([[49, 2], [12, 94]])
    print(DensityMatrices.fuzz(A, B))
