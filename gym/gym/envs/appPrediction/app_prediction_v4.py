import logging
import gym
from gym import spaces
from gym.utils import seeding
from six import StringIO
import sys
import numpy as np
import util
logger = logging.getLogger(__name__)

class AppPredictionV4(gym.Env):
    metadata = {
        'render.modes': ['human', 'ansi'],
        'video.frames_per_second': 60
    }
    def __init__(self, properties):
        self.__init_done = False
        self.__data = None
        self.properties = util.register(properties)
        self._configure()
        self._seed()
        sample_app = self.__data.get_one(type="apps")
        embeddings = np.array(sample_app['embeddings']).ravel()
        self.__emb_size = embeddings.size
        other_features = np.array(sample_app['other_feat']).ravel()
        self.__other_size = other_features.size
        self.user_bounds = self.__data.get_boundary(type="ratings")
        self.user_exploration = np.arange(self.user_bounds[0], self.user_bounds[1] + 1)
        self.np_random.shuffle(self.user_exploration)
        self.__previous_user = None
        self.__current_user = None
        self.obs_id = sample_app["_id"]
        self.__embeddings = np.ones(shape=self.__emb_size)
        self.__other_feat = np.ones(shape=self.__other_size)
        self.__true_positives = self.__true_negatives = self.__false_positives = self.__false_negatives = 0.
        self.__selected_apps = []
        low = np.zeros(self.__emb_size)
        high = np.ones(self.__emb_size)
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.SimBox(low, high, env=self, top_n=self.properties.kwargs['expl_subset_limit'])
        self.viewer = None
        self.__init_done = True

    def _configure(self, display=None):
        if not self.__init_done:
            self.__data = util.LocalFileData(kwargs=self.properties.kwargs)
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_data(self):
        return self.__data

    def _reset(self):
        if len(self.user_exploration) == 0:
            self.user_exploration = np.arange(self.user_bounds[0], self.user_bounds[1] + 1)
        self.__current_user = self.user_exploration[0]
        self.user_exploration = self.user_exploration[1:]
        self.__selected_apps = []
        self.obs_id, app = self._get_guided_random_app()
        self.__embeddings = np.array(app['embeddings']).ravel()
        self.__other_feat = np.array(app['other_feat']).ravel()
        self.__true_positives = self.__true_negatives = self.__false_positives = self.__false_negatives = 0.
        return self._get_obs()

    def _get_obs(self):
        return self.__embeddings

    def _get_guided_random_app(self):
        subset_size = self.properties.kwargs['expl_subset_limit']
        user_apps = 1 + self.__data.get_user_apps(self.__current_user, limit=subset_size)
        random_app = int(np.floor(self.np_random.uniform(low=0, high=subset_size)))
        random_app_id = user_apps[random_app]
        return random_app_id, self.__data.get_one(query=random_app_id, type="apps")

    def _step(self, action):
        done = False
        is_chosen, p_action, pred_rating, action_id = self.__data.compute_transition(self.__current_user, self.obs_id, action, embeddings=(self.__emb_size, self.__other_size))
        if is_chosen:
            self.__true_positives += 1
            p_end_episode = 0.1
            action_id = int(action_id)
            app_info = {"previous": self.obs_id, "app": action_id, "reward": pred_rating, "correct": "^"}
        else:
            self.__false_positives += 1
            p_end_episode = 0.2
            if self.properties.kwargs["guided_exploration"]:
                _, random_app = self._get_guided_random_app()
            else:
                random_app = self.__data.get_one(type="apps")
            action_id = random_app["_id"]
            app_info = {"previous": self.obs_id, "app": action_id, "reward": 0., "correct": "x"}
            pred_rating = 0.
        reward = pred_rating
        self.__selected_apps.append(app_info)
        if self.np_random.uniform() < p_end_episode:
            done = True
            self.__previous_user = self.__current_user
        self.obs_id = action_id
        _, self.__embeddings, self.__other_feat = self.__data.get_app_representation(self.obs_id)
        info = {"None"}
        return self._get_obs(), round(reward), done, info

    def _render(self, mode='human', close=False):
        if close:
            return
        output = StringIO() if mode == 'ansi' else sys.stdout
        head = "User: {}\n".format(self.__previous_user)
        head += "\n"
        output.write(head)
        body = "  #  ||  Prev App  || Now App || Reward || Correct \n"
        i = 0
        reward = 0.
        for app in self.__selected_apps:
            reward += app["reward"]
            body += "{0:05d}||   {1:06d}   || {2:06d}  ||  {3:.2f}  ||___{4}___ \n".format(i, app["previous"], app["app"], app["reward"], app["correct"])
            i += 1
        output.write(body)
        precision = self.__true_positives / (self.__true_positives + self.__false_positives) if (self.__true_positives + self.__false_positives) > 0 else 0.
        stats = "Precision  : {0:.3f}\n".format(precision)
        output.write(stats)
        return output
