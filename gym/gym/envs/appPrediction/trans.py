import numpy as np
from scipy.sparse import csr_matrix
import os


class Transition(object):
    def __init__(self, shape, raw_data=None, data_dir='data'):
        self.__shape = shape
        self.__raw_data = raw_data
        self.apps_collection_filename = os.path.join(data_dir, "apps.npy")
        self.rating_matrix_filename = os.path.join(data_dir, "usingapp.npy")
        self.similarity_matrix_filename = os.path.join(data_dir, "similar.npy")
        self.transition_matrix_filename = os.path.join(data_dir, "trans.npy")
        self.ranking_matrix_filename = os.path.join(data_dir, "ranking.npy")
        self.__ratings_matrix = np.load(self.rating_matrix_filename, "r", allow_pickle=False)
        self.__sim_matrix = np.load(self.similarity_matrix_filename, "r", allow_pickle=False)
        self.__tr_matrix = np.load(self.transition_matrix_filename, "r", allow_pickle=False)
        self.__ranking_matrix = np.load(self.ranking_matrix_filename, "r", allow_pickle=False)

    def get_transition_matrix(self, beta=0.9):
        sum_Sim = np.sum(self.__sim_matrix, axis=1)
        self.__tr_matrix = np.zeros_like(self.__sim_matrix)
        for i in range(self.__sim_matrix.shape[1]):
            a = self.__sim_matrix[i, ...]
            b = sum_Sim[i]
            c = (((beta * a) / b) if sum_Sim[i] > 0 else 0)
            d = self.__sim_matrix.shape[1]
            self.__tr_matrix[i, ...] = c + ((1. - beta) / d)
        return self.__tr_matrix

    def get_transitions_per_app(self, app_id):
        return self.get_transition_matrix()[app_id - 1]

    def _compute_rankings(self, alpha=0.1, scale=False):
        random_walk_length = csr_matrix(1 - alpha * self.__tr_matrix).todense()
        inv_random_walk_length = np.linalg.pinv(random_walk_length)
        P_hat = alpha * self.__tr_matrix * csr_matrix(inv_random_walk_length)
        self.__ranking_matrix = self.__ratings_matrix * csr_matrix(P_hat)
        return self._scale_rows(self.__ranking_matrix) if scale else self.__ranking_matrix.toarray()

    def predict_app_rating(self, user_id, app_id):
        pred_cosine = 0.
        if np.count_nonzero(self.__ratings_matrix[:, app_id - 1]):
            sim_cosine = self.__sim_matrix[app_id - 1]
            ind = (self.__ratings_matrix[user_id - 1] > 0)
            normal_cosine = np.sum(np.absolute(sim_cosine[ind]))
            if normal_cosine > 0:
                pred_cosine = np.dot(sim_cosine, self.__ratings_matrix[user_id - 1]) / normal_cosine
        return np.clip(pred_cosine, 0, 5)

    def _scale_rows(self, matrix):
        m = matrix.toarray()
        max_indexes = np.argmax(m, axis=1)
        scales = [5. / m[crow][max_indexes[crow]] for crow in np.arange(len(max_indexes))]
        scales = np.expand_dims(scales, axis=1)
        return np.round(scales * matrix.toarray())

    def get_rankings_per_user(self, user_id):
        return self.__ranking_matrix[user_id - 1]


def readingFile(filename, split='\t'):
    f = open(filename, "r")
    data = []
    for row in f:
        r = row.split(split)
        e = [int(r[0]), int(r[1]), int(r[2])]
        data.append(e)
    return data
