
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

class PairwiseClustering:
    def __init__(
        self,
        n_clusters=4,
        n_samples=20,
    ):
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        
    def fit(self, X, y=None):
        
        assert self.n_samples >= 2
        assert X.shape[0] == self.n_clusters * self.n_samples
        
        labels = [[i] for i in range(self.n_clusters)]
        sims = []
        for i in range(self.n_samples-1):
            from_idx = list(range(i, X.shape[0], self.n_samples))
            to_idx = list(range(i+1, X.shape[0], self.n_samples))
            
            from_data = X[from_idx]
            to_data = X[to_idx]
            sim = cosine_similarity(from_data, to_data)
            sims.append(sim)
            row_ind, col_ind = linear_sum_assignment(1-sim)
            match_map = dict(zip(row_ind, col_ind))
            # update
            for k in range(self.n_clusters):
                labels[k].append(match_map[labels[k][-1]])
        
        labels = [item for sublist in labels for item in sublist]
        self.labels_ = labels
        self.sims_ = sims
        
        return self
     
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 