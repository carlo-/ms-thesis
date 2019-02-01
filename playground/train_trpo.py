import os

import numpy as np
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import TRPO

import gym
from gym.wrappers import FlattenDictWrapper
from gym.envs.robotics import FetchPickAndPlaceEnv
from gym.envs.robotics.fetch_env import goal_distance as fetch_env_goal_distance


TRAIN = True
OUT_DIR = '../out'
TB_DIR = f'{OUT_DIR}/tensorboard3'
os.makedirs(TB_DIR, exist_ok=True)


def num_to_short_str(x: int) -> str:
    if x < 1_000:
        return str(x)
    elif 1_000 <= x < 1_000_000:
        return f'{x // 1_000}k'
    elif 1_000_000 <= x < 1_000_000_000:
        return f'{x // 1_000_000}M'
    else:
        return str(x)


def init_env(seed=0, reward_params=None):
    def _init():
        env = gym.make('FetchPickAndPlaceDense-v1')
        raw_env = env.unwrapped # type: FetchPickAndPlaceEnv
        raw_env.reward_params = reward_params
        env.seed(seed)
        env = FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
        return env
    return _init


def train(reward_params, trpo_params, steps, model_path, seed=42, tb_log_name='TRPO',
          normalize_obs=False, vec_normalize_dir=None, load_path_first=None):

    env = DummyVecEnv([init_env(seed=seed, reward_params=reward_params)])

    if normalize_obs:
        raise NotImplementedError('Untested')
        assert vec_normalize_dir is not None
        os.makedirs(vec_normalize_dir, exist_ok=True)
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
        raw_env = env.venv.envs[0].unwrapped # type: FetchPickAndPlaceEnv
    else:
        raw_env = env.envs[0].unwrapped # type: FetchPickAndPlaceEnv

    assert isinstance(raw_env, FetchPickAndPlaceEnv)

    def callback(model_locals, model_globals):

        step = model_locals.get('timesteps_so_far') # type: int
        writer = model_locals.get('writer') # type: tf.summary.FileWriter

        if normalize_obs:
            obs = env.old_obs[0].copy()
        else:
            obs = model_locals.get('observation')

        if obs is None or len(obs) == 0:
            return

        achieved_goals, goals = obs[..., 3:6], obs[..., -3:]
        original_rewards = -fetch_env_goal_distance(achieved_goals, goals)
        avg_original_rew = np.mean(original_rewards)

        s = tf.Summary()
        s.value.add(tag='original_episode_reward', simple_value=avg_original_rew)
        writer.add_summary(s, step)

    if load_path_first is not None and os.path.exists(load_path_first):
        print('Loading existing model first!')
        model = TRPO.load(load_path_first, env, verbose=1, tensorboard_log=TB_DIR, **trpo_params)
    else:
        model = TRPO(MlpPolicy, env, verbose=1, tensorboard_log=TB_DIR, **trpo_params)
    model.learn(total_timesteps=steps, tb_log_name=tb_log_name, callback=callback)
    model.save(model_path)

    if normalize_obs:
        env.save_running_average(vec_normalize_dir)


def train_configs(config_index=None, play_only=False, steps=0, load_with_suffix=None):
    assert steps > 0
    configs = [
        dict(
            name='k1_c0.5_md0.03_gb2',
            reward_params=dict(k=1.0, c=0.5, min_dist=0.03, grasp_bonus=2.0)
        ),
        dict(
            name='k1_c1_md0.03_gb2',
            reward_params=dict(k=1.0, c=1.0, min_dist=0.03, grasp_bonus=2.0)
        ),
        dict(
            name='k1_c0.5_md0.03_gb0',
            reward_params=dict(k=1.0, c=0.5, min_dist=0.03, grasp_bonus=0.0)
        ),
        dict(
            name='k1_c1_md0.03_gb0',
            reward_params=dict(k=1.0, c=1.0, min_dist=0.03, grasp_bonus=0.0)
        )
    ]

    if config_index is not None:
        configs = [configs[config_index]]

    for conf in configs:
        old_model_name = f'{OUT_DIR}/fetch_trpo_model_{conf["name"]}{load_with_suffix or ""}'

        load_path_first = None
        if isinstance(load_with_suffix, str):
            load_path_first = old_model_name + '.pkl'

        model_name = f'{old_model_name}_s{num_to_short_str(steps)}'
        model_path = model_name + '.pkl'
        tb_name = f'TRPO_{conf["name"]}'
        seed = 42
        if play_only:
            play(model_path, conf['reward_params'])
        else:
            normalize_obs = conf.get('normalize_obs', False)
            vec_normalize_dir = f'{OUT_DIR}/fetch_trpo_vec_norm_{conf["name"]}_s5M'
            if normalize_obs:
                os.makedirs(vec_normalize_dir, exist_ok=True)
            train(conf['reward_params'], conf.get('trpo_params', dict()), steps, model_path, seed, tb_log_name=tb_name,
                  normalize_obs=normalize_obs, vec_normalize_dir=vec_normalize_dir, load_path_first=load_path_first)


def play(model_path, reward_params=None):
    env = DummyVecEnv([init_env(seed=42, reward_params=reward_params)])
    model = TRPO.load(model_path, env=env)
    while True:
        obs = env.reset()
        for _ in range(100):
            env.render()
            u = model.predict(obs, deterministic=True)[0]
            obs, rew, done, info = env.step(u)
            print(rew)


if __name__ == '__main__':
    train_configs(3, steps=2_000_000, load_with_suffix='_s1M_cont')
