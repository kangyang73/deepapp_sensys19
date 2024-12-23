import atexit
import logging
import json
import numpy as np
import os
import six
import sys
import threading
import weakref

from gym import error, version
from gym.monitoring import stats_recorder, video_recorder
from gym.utils import atomic_write, closer, seeding

logger = logging.getLogger(__name__)

FILE_PREFIX = 'openaigym'
MANIFEST_PREFIX = FILE_PREFIX + '.manifest'

def detect_training_manifests(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith(MANIFEST_PREFIX + '.')]

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith(FILE_PREFIX + '.')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return

    logger.info('Clearing %d monitor files from previous run (because force=True was provided)', len(files))
    for file in files:
        os.unlink(file)

def capped_cubic_video_schedule(episode_id):
    if episode_id < 1000:
        return int(round(episode_id ** (1. / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0

def disable_videos(episode_id):
    return False

monitor_closer = closer.Closer()

# This method gets used for a sanity check in scoreboard/api.py. It's
# not intended for use outside of the gym codebase.
def _open_monitors():
    return list(monitor_closer.closeables.values())

class Monitor(object):
    """A configurable monitor for your training runs.

    Every env has an attached monitor, which you can access as
    'env.monitor'. Simple usage is just to call 'monitor.start(dir)'
    to begin monitoring and 'monitor.close()' when training is
    complete. This will record stats and will periodically record a video.

    For finer-grained control over how often videos are collected, use the
    video_callable argument, e.g.
    'monitor.start(video_callable=lambda count: count % 100 == 0)'
    to record every 100 episodes. ('count' is how many episodes have completed)

    Depending on the environment, video can slow down execution. You
    can also use 'monitor.configure(video_callable=lambda count: False)' to disable
    video.

    Monitor supports multiple threads and multiple processes writing
    to the same directory of training data. The data will later be
    joined by scoreboard.upload_training_data and on the server.

    Args:
        env (gym.Env): The environment instance to monitor.

    Attributes:
        id (Optional[str]): The ID of the monitored environment

    """

    def __init__(self, env):
        # Python's GC allows refcycles *or* for objects to have a
        # __del__ method. So we need to maintain a weakref to env.
        #
        # https://docs.python.org/2/library/gc.html#gc.garbage
        self._env_ref = weakref.ref(env)
        self.videos = []

        self.stats_recorder = None
        self.video_recorder = None
        self.enabled = False
        self.episode_id = 0
        self._monitor_id = None
        self.seeds = None

    @property
    def env(self):
        env = self._env_ref()
        if env is None:
            raise error.Error("env has been garbage collected. To keep using a monitor, you must keep around a reference to the env object. (HINT: try assigning the env to a variable in your code.)")
        return env

    def start(self, directory, video_callable=None, force=False, resume=False, seed=None):
        """Start monitoring.

        Args:
            directory (str): A per-training run directory where to record stats.
            video_callable (Optional[function, False]): function that takes in the index of the episode and outputs a boolean, indicating whether we should record a video on this episode. The default (for video_callable is None) is to take perfect cubes, capped at 1000. False disables video recording.
            force (bool): Clear out existing training data from this directory (by deleting every file prefixed with "openaigym.").
            resume (bool): Retain the training data already in this directory, which will be merged with our new data
            seed (Optional[int]): The seed to run this environment with. By default, a random seed will be chosen.
        """
        if self.env.spec is None:
            logger.warn("Trying to monitor an environment which has no 'spec' set. This usually means you did not create it via 'gym.make', and is recommended only for advanced users.")

        if not os.path.exists(directory):
            logger.info('Creating monitor directory %s', directory)
            os.makedirs(directory)

        if video_callable is None:
            video_callable = capped_cubic_video_schedule
        elif video_callable == False:
            video_callable = disable_videos
        elif not callable(video_callable):
            raise error.Error('You must provide a function, None, or False for video_callable, not {}: {}'.format(type(video_callable), video_callable))

        # Check on whether we need to clear anything
        if force:
            clear_monitor_files(directory)
        elif not resume:
            training_manifests = detect_training_manifests(directory)
            if len(training_manifests) > 0:
                raise error.Error('''Trying to write to monitor directory {} with existing monitor files: {}.

 You should use a unique directory for each training run, or use 'force=True' to automatically clear previous monitor files.'''.format(directory, ', '.join(training_manifests[:5])))


        self._monitor_id = monitor_closer.register(self)

        self.enabled = True
        self.directory = os.path.abspath(directory)
        # We use the 'openai-gym' prefix to determine if a file is
        # ours
        self.file_prefix = FILE_PREFIX
        self.file_infix = '{}.{}'.format(self._monitor_id, os.getpid())
        self.stats_recorder = stats_recorder.StatsRecorder(directory, '{}.episode_batch.{}'.format(self.file_prefix, self.file_infix))
        self.configure(video_callable=video_callable)
        if not os.path.exists(directory):
            os.mkdir(directory)

        seeds = self.env.seed(seed)
        self.seeds = seeds

    def flush(self):
        """Flush all relevant monitor information to disk."""
        self.stats_recorder.flush()

        # Give it a very distiguished name, since we need to pick it
        # up from the filesystem later.
        path = os.path.join(self.directory, '{}.manifest.{}.manifest.json'.format(self.file_prefix, self.file_infix))
        logger.debug('Writing training manifest file to %s', path)
        with atomic_write.atomic_write(path) as f:
            # We need to write relative paths here since people may
            # move the training_dir around. It would be cleaner to
            # already have the basenames rather than basename'ing
            # manually, but this works for now.
            json.dump({
                'stats': os.path.basename(self.stats_recorder.path),
                'videos': [(os.path.basename(v), os.path.basename(m))
                           for v, m in self.videos],
                'env_info': self._env_info(),
                'seeds': self.seeds,
            }, f)

    def close(self):
        """Flush all monitor data to disk and close any open rending windows."""
        if not self.enabled:
            return
        self.stats_recorder.close()
        if self.video_recorder is not None:
            self._close_video_recorder()
        self.flush()

        env = self._env_ref()
        # Only take action if the env hasn't been GC'd
        if env is not None:
            # Note we'll close the env's rendering window even if we did
            # not open it. There isn't a particular great way to know if
            # we did, since some environments will have a window pop up
            # during video recording.
            try:
                env.render(close=True)
            except Exception as e:
                if env.spec:
                    key = env.spec.id
                else:
                    key = env
                # We don't want to avoid writing the manifest simply
                # because we couldn't close the renderer.
                logger.error('Could not close renderer for %s: %s', key, e)

            # Remove the env's pointer to this monitor
            del env._monitor

        # Stop tracking this for autoclose
        monitor_closer.unregister(self._monitor_id)
        self.enabled = False

        logger.info('''Finished writing results. You can upload them to the scoreboard via gym.upload(%r)''', self.directory)

    def configure(self, video_callable=None):
        """Reconfigure the monitor.

            video_callable (function): Whether to record video to upload to the scoreboard.
        """

        if video_callable is not None:
            self.video_callable = video_callable

    def _before_step(self, action):
        if not self.enabled: return
        self.stats_recorder.before_step(action)

    def _after_step(self, observation, reward, done, info):
        if not self.enabled: return done

        # Add 1 since about to take another step
        if self.env.spec and self.stats_recorder.steps+1 >= self.env.spec.timestep_limit:
            logger.info('Ending episode %i because it reached the timestep limit of %i.', self.episode_id, self.env.spec.timestep_limit)
            done = True

        # Record stats
        self.stats_recorder.after_step(observation, reward, done, info)
        # Record video
        self.video_recorder.capture_frame()

        return done


    def _before_reset(self):
        if not self.enabled: return
        self.stats_recorder.before_reset()

    def _after_reset(self, observation):
        if not self.enabled: return

        # Reset the stat count
        self.stats_recorder.after_reset(observation)

        # Close any existing video recorder
        if self.video_recorder:
            self._close_video_recorder()

        # Start recording the next video.
        #
        # TODO: calculate a more correct 'episode_id' upon merge
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=os.path.join(self.directory, '{}.video.{}.video{:06}'.format(self.file_prefix, self.file_infix, self.episode_id)),
            metadata={'episode_id': self.episode_id},
            enabled=self._video_enabled(),
        )
        self.video_recorder.capture_frame()

        # Bump *after* all reset activity has finished
        self.episode_id += 1

        self.flush()

    def _close_video_recorder(self):
        self.video_recorder.close()
        if self.video_recorder.functional:
            self.videos.append((self.video_recorder.path, self.video_recorder.metadata_path))

    def _video_enabled(self):
        return self.video_callable(self.episode_id)

    def _env_info(self):
        env_info = {
            'gym_version': version.VERSION,
        }
        if self.env.spec:
            env_info['env_id'] = self.env.spec.id
        return env_info

    def __del__(self):
        # Make sure we've closed up shop when garbage collecting
        self.close()

def load_results(training_dir):
    if not os.path.exists(training_dir):
        return

    manifests = detect_training_manifests(training_dir)
    if not manifests:
        return

    logger.debug('Uploading data from manifest %s', ', '.join(manifests))

    # Load up stats + video files
    stats_files = []
    videos = []
    main_seeds = []
    seeds = []
    env_infos = []

    for manifest in manifests:
        with open(manifest) as f:
            contents = json.load(f)
            # Make these paths absolute again
            stats_files.append(os.path.join(training_dir, contents['stats']))
            videos += [(os.path.join(training_dir, v), os.path.join(training_dir, m))
                       for v, m in contents['videos']]
            env_infos.append(contents['env_info'])
            current_seeds = contents.get('seeds', [])
            seeds += current_seeds
            if current_seeds:
                main_seeds.append(current_seeds[0])
            else:
                # current_seeds could be None or []
                main_seeds.append(None)

    env_info = collapse_env_infos(env_infos, training_dir)
    timestamps, episode_lengths, episode_rewards, initial_reset_timestamp = merge_stats_files(stats_files)

    return {
        'manifests': manifests,
        'env_info': env_info,
        'timestamps': timestamps,
        'episode_lengths': episode_lengths,
        'episode_rewards': episode_rewards,
        'initial_reset_timestamp': initial_reset_timestamp,
        'videos': videos,
        'main_seeds': main_seeds,
        'seeds': seeds,
    }

def merge_stats_files(stats_files):
    timestamps = []
    episode_lengths = []
    episode_rewards = []
    initial_reset_timestamps = []

    for path in stats_files:
        with open(path) as f:
            content = json.load(f)
            timestamps += content['timestamps']
            episode_lengths += content['episode_lengths']
            episode_rewards += content['episode_rewards']
            initial_reset_timestamps.append(content['initial_reset_timestamp'])

    idxs = np.argsort(timestamps)
    timestamps = np.array(timestamps)[idxs].tolist()
    episode_lengths = np.array(episode_lengths)[idxs].tolist()
    episode_rewards = np.array(episode_rewards)[idxs].tolist()
    initial_reset_timestamp = min(initial_reset_timestamps)
    return timestamps, episode_lengths, episode_rewards, initial_reset_timestamp

def collapse_env_infos(env_infos, training_dir):
    assert len(env_infos) > 0

    first = env_infos[0]
    for other in env_infos[1:]:
        if first != other:
            raise error.Error('Found two unequal env_infos: {} and {}. This usually indicates that your training directory {} has commingled results from multiple runs.'.format(first, other, training_dir))

    for key in ['env_id', 'gym_version']:
        if key not in first:
            raise error.Error("env_info {} from training directory {} is missing expected key {}. This is unexpected and likely indicates a bug in gym.".format(first, training_dir, key))
    return first
