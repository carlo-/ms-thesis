import os

import numpy as np
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2

import gym
from gym.wrappers import FlattenDictWrapper
from gym.envs.robotics import FetchPickAndPlaceEnv
from gym.envs.robotics.fetch_env import goal_distance as fetch_env_goal_distance


TRAIN = True
OUT_DIR = '../out'
TB_DIR = f'{OUT_DIR}/tensorboard4'
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


def train(reward_params, ppo_params, steps, model_path, seed=42, tb_log_name='PPO2',
          load_path_first=None, n_cpu=1):

    # env = DummyVecEnv([init_env(seed=seed, reward_params=reward_params)])
    env = SubprocVecEnv([init_env(seed=seed+i, reward_params=reward_params) for i in range(n_cpu)])

    if load_path_first is not None and os.path.exists(load_path_first):
        print('Loading existing model first!')
        model = PPO2.load(load_path_first, env, verbose=1, tensorboard_log=TB_DIR, **ppo_params)
    else:
        model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=TB_DIR, **ppo_params)

    def callback(model_locals, model_globals):

        writer = model_locals.get('writer') # type: tf.summary.FileWriter
        update = model_locals.get('update') # type: int
        step = update * model.n_batch

        obs = model_locals.get('obs')
        if obs is None or len(obs) == 0:
            return

        achieved_goals, goals = obs[..., 3:6], obs[..., -3:]
        original_rewards = -fetch_env_goal_distance(achieved_goals, goals)
        avg_original_rew = np.mean(original_rewards)

        s = tf.Summary()
        s.value.add(tag='original_episode_reward', simple_value=avg_original_rew)
        writer.add_summary(s, step)

    model.learn(total_timesteps=steps, tb_log_name=tb_log_name, callback=callback)
    model.save(model_path)


def train_configs(config_index=None, play_only=False, steps=0, load_with_suffix=None, n_cpu=1):
    assert steps > 0
    configs = [
        dict(
            name='k1_c0.5_md0.03_gb2',
            reward_params=dict(k=1.0, c=0.5, min_dist=0.03, grasp_bonus=2.0)
        ),
        dict(
            name='original_rew',
            reward_params=None
        )
    ]

    if config_index is not None:
        configs = [configs[config_index]]

    for conf in configs:
        old_model_name = f'{OUT_DIR}/fetch_ppo2_model_{conf["name"]}{load_with_suffix or ""}'

        load_path_first = None
        if isinstance(load_with_suffix, str):
            load_path_first = old_model_name + '.pkl'

        model_name = f'{old_model_name}_s{num_to_short_str(steps)}'
        model_path = model_name + '.pkl'
        tb_name = f'PPO2_{conf["name"]}'
        seed = 42
        if play_only:
            play(load_path_first, conf['reward_params'])
        else:
            train(conf['reward_params'], conf.get('ppo_params', dict()), steps, model_path, seed, tb_log_name=tb_name,
                  load_path_first=load_path_first, n_cpu=n_cpu)


def play(model_path, reward_params=None):
    env = DummyVecEnv([init_env(seed=42, reward_params=reward_params)])
    model = PPO2.load(model_path, env=env)
    while True:
        obs = env.reset()
        for _ in range(100):
            env.render()
            u = model.predict(obs, deterministic=True)[0]
            obs, rew, done, info = env.step(u)
            print(rew)


if __name__ == '__main__':
    train_configs(0, steps=10_000_000, n_cpu=7, load_with_suffix='_s10M', play_only=True)
