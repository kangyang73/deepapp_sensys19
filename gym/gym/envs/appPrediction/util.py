import os
import pandas
from trans import Transition
import numpy as np


class LocalFileData(object):
    def __init__(self, kwargs):
        self.local_properties = kwargs['local']
        path = self.local_properties['path']
        self.data_dir = path
        users, apps, records = self._get_info(os.path.join(path, self.local_properties['info']))
        self.__stats = {"users": users, "apps": apps,  "records": records}
        apps_file = self.local_properties['apps_collection']
        self.__apps = self._read_npy(os.path.join(path, apps_file))
        self.__user_records = self._read_npy(os.path.join(self.data_dir, self.local_properties['records_collection']))
        self.__transitionmatrix = Transition(raw_data=self.__user_records, shape=(users, apps), data_dir=self.data_dir)

    def _read_csv(self, file_path, separator=','):
        return pandas.read_csv(file_path, sep=separator)

    def _read_npy(self, filename):
        return np.load(filename, "r", allow_pickle=False)

    def _get_info(self, file_path):
        f = open(file_path, "r")
        users = apps = records = None
        i = 0
        for row in f:
            if i == 0:
                users = int(row.split(" ")[0])
            elif i == 1:
                apps = int(row.split(" ")[0])
            elif i == 2:
                records = int(row.split(" ")[0])
                break
            i += 1
        return users, apps, records

    def get_boundary(self, type="apps"):
        if type == "apps":
            return 0, self.__apps.shape[0]
        else:
            return 1, self.__user_records.shape[0]

    def get_one(self, query=None, type="apps"):
        if type == "apps":
            app = {}
            if query is None:
                rows = self.__apps.shape[0] - 1
                app["_id"] = int(np.round(np.random.uniform(0, rows)))
            else:
                app["_id"] = query - 1
            app["embeddings"] = self.__apps[app["_id"]]
            app['other_feat'] = []
            return app
        else:
            rating = None
            if query is None:
                rows = self.__user_records.shape[0] - 1
                rating = self.__user_records[np.round(np.random.uniform(0, rows))]
            else:
                rating = rating
            return rating

    def get_user_apps(self, user_id, limit=5):
        return self.__user_records[user_id - 1].argsort()[::-1][:limit]

    def get_app_representation(self, app_id):
        query = app_id
        app = self.get_one(query=query, type="apps")
        return self.construct_app_representation(app)

    def compute_transition(self, user_id, prev_obs, obs, embeddings=None):
        if embeddings is not None:
            rs = np.all(self.__apps == obs, axis=1)
            obs_id = 1 + np.argmax(rs)
        else:
            obs_id = obs
        app_transitions = self.__transitionmatrix.get_transitions_per_app(prev_obs)
        max = np.mean(app_transitions)
        std = np.std(app_transitions)
        threshold = max - std
        chosen = app_transitions[obs_id - 1] > threshold
        rating = self.__user_records[user_id - 1][obs_id - 1]
        rating = rating if rating > 0 else self.__transitionmatrix.predict_app_rating(user_id, obs_id)
        return chosen and rating > 0, app_transitions[obs_id - 1], np.round(rating), obs_id

    def construct_app_representation(self, app):
        embeddings = np.array(app['embeddings']).ravel()
        other_feat = np.array(app['other_feat']).ravel()
        return np.concatenate([embeddings, other_feat]).ravel(), embeddings, other_feat

    def get_likely_apps_per_state(self, state_id, top_n=20):
        app_transitions = self.__transitionmatrix.get_transitions_per_app(state_id - 1)
        top_n_apps = app_transitions.argsort()[::-1][:top_n]
        return 1 + top_n_apps


class register():
    def __init__(self, properties):
        self.kwargs = properties