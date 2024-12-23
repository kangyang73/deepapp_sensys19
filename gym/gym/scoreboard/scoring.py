"""This is the actual code we use to score people's solutions
server-side. The interfaces here are not yet stable, but we include
them so that people can reproduce our scoring calculations
independently.

We correspondly do not currently import this module.
"""

import numpy as np
import requests

import gym

def score_from_remote(url):
    result = requests.get(url)
    parsed = result.json()
    episode_lengths = parsed['episode_lengths']
    episode_rewards = parsed['episode_rewards']
    timestamps = parsed['timestamps']
    # Handle legacy entries where initial_reset_timestamp wasn't set
    initial_reset_timestamp = parsed.get('initial_reset_timestamp', timestamps[0])
    env_id = parsed['env_id']

    spec = gym.spec(env_id)
    return score_from_merged(episode_lengths, episode_rewards, timestamps, initial_reset_timestamp, spec.trials, spec.reward_threshold)

def score_from_local(directory):
    """Calculate score from a local results directory"""
    results = gym.monitoring.monitor.load_results(directory)
    # No scores yet saved
    if results is None:
        return None

    episode_lengths = results['episode_lengths']
    episode_rewards = results['episode_rewards']
    timestamps = results['timestamps']
    initial_reset_timestamp = results['initial_reset_timestamp']
    spec = gym.spec(results['env_info']['env_id'])

    return score_from_merged(episode_lengths, episode_rewards, timestamps, initial_reset_timestamp, spec.trials, spec.reward_threshold)

def score_from_merged(episode_lengths, episode_rewards, timestamps, initial_reset_timestamp, trials, reward_threshold):
    """Method to calculate the score from merged monitor files.
    """
    # Make sure everything is a float -- no pesky ints.
    episode_rewards = np.array(episode_rewards, dtype='float64')

    episode_t_value = timestep_t_value = mean = error = None
    seconds_to_solve = seconds_in_total = None

    if len(timestamps) > 0:
        # This is: time from the first reset to the end of the last episode
        seconds_in_total = timestamps[-1] - initial_reset_timestamp
    if len(episode_rewards) >= trials:
        means = running_mean(episode_rewards, trials)
        if reward_threshold is not None:
            # Compute t-value by finding the first index at or above
            # the threshold. It comes out as a singleton tuple.
            (indexes_above_threshold, ) = np.where(means >= reward_threshold)
            if len(indexes_above_threshold) > 0:
                # Grab the first episode index that is above the threshold value
                episode_t_value = indexes_above_threshold[0]

                # Find timestep corresponding to this episode
                cumulative_timesteps = np.cumsum(np.insert(episode_lengths, 0, 0))
                # Convert that into timesteps
                timestep_t_value = cumulative_timesteps[episode_t_value]
                # This is: time from the first reset to the end of the first solving episode
                seconds_to_solve = timestamps[episode_t_value] - initial_reset_timestamp

        # Find the window with the best mean
        best_idx = np.argmax(means)
        best_rewards = episode_rewards[best_idx:best_idx+trials]
        mean = np.mean(best_rewards)
        if trials == 1: # avoid NaN
            error = 0.
        else:
            error = np.std(best_rewards) / (np.sqrt(trials) - 1)
    return {
        'episode_t_value': episode_t_value,
        'timestep_t_value': timestep_t_value,
        'mean': mean,
        'error': error,
        'number_episodes': len(episode_rewards),
        'number_timesteps': sum(episode_lengths),
        'seconds_to_solve': seconds_to_solve,
        'seconds_in_total': seconds_in_total,
    }

def running_mean(x, N):
    x = np.array(x, dtype='float64')
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

def compute_graph_stats(episode_lengths, episode_rewards, timestamps, initial_reset_timestamp, buckets):
    """Method to compute the aggregates for the graphs."""
    # Not a dependency of OpenAI Gym generally.
    import scipy.stats

    num_episodes = len(episode_lengths)

    # Catch for if no files written which causes error with scipy.stats.binned_statistic
    if num_episodes == 0:
        return None

    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)

    # The index of the start of each episode
    x_timestep = np.cumsum(np.insert(episode_lengths, 0, 0))[:-1]
    assert len(x_timestep) == num_episodes

    # Delta since the beginning of time
    x_seconds = [timestamp - initial_reset_timestamp for timestamp in timestamps]

    # The index of each episode
    x_episode = range(num_episodes)

    # Calculate the appropriate x/y statistics
    x_timestep_y_reward = scipy.stats.binned_statistic(x_timestep, episode_rewards, 'median', buckets)
    x_timestep_y_length = scipy.stats.binned_statistic(x_timestep, episode_lengths, 'median', buckets)

    x_episode_y_reward = scipy.stats.binned_statistic(x_episode, episode_rewards, 'median', buckets)
    x_episode_y_length = scipy.stats.binned_statistic(x_episode, episode_lengths, 'median', buckets)

    x_seconds_y_reward = scipy.stats.binned_statistic(x_seconds, episode_rewards, 'median', buckets)
    x_seconds_y_length = scipy.stats.binned_statistic(x_seconds, episode_lengths, 'median', buckets)

    return {
        'initial_reset_timestamp': initial_reset_timestamp,
        'x_timestep_y_reward': graphable_binned_statistic(x_timestep_y_reward),
        'x_timestep_y_length': graphable_binned_statistic(x_timestep_y_length),
        'x_episode_y_reward': graphable_binned_statistic(x_episode_y_reward),
        'x_episode_y_length': graphable_binned_statistic(x_episode_y_length),
        'x_seconds_y_length': graphable_binned_statistic(x_seconds_y_length),
        'x_seconds_y_reward': graphable_binned_statistic(x_seconds_y_reward),
    }

def graphable_binned_statistic(binned):
    x = running_mean(binned.bin_edges, 2)
    y = binned.statistic
    assert len(x) == len(y)

    # Get rid of nasty NaNs
    valid = np.logical_not(np.isnan(x)) & np.logical_not(np.isnan(y))
    x = x[valid]
    y = y[valid]

    return {
        'x': x,
        'y': y,
    }
