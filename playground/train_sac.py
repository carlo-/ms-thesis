import os

import tensorflow as tf

from stable_baselines.bench import Monitor
from stable_baselines.sac import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC

import gym
from gym.wrappers import FlattenDictWrapper
from gym.envs.robotics import FetchPickAndPlaceEnv
from gym.envs.robotics.fetch_env import goal_distance as fetch_env_goal_distance


TRAIN = True
OUT_DIR = '../out'


def init_env(seed=0, reward_params=None):
    def _init():
        env = gym.make('FetchPickAndPlaceDense-v1')
        env.unwrapped.reward_params = reward_params
        env.seed(seed)
        env = FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
        return env
    return _init


def train(reward_params, sac_params, steps, model_path, seed=42, tb_log_name='SAC'):

    tb_dir = f'{OUT_DIR}/tensorboard/'
    os.makedirs(tb_dir, exist_ok=True)

    env = DummyVecEnv([init_env(seed=seed, reward_params=reward_params)])
    raw_env = env.unwrapped # type: FetchPickAndPlaceEnv
    assert isinstance(raw_env, FetchPickAndPlaceEnv)
    original_cumulative_rew = 0.0

    def callback(sac_locals, sac_globals):
        global original_cumulative_rew

        step = sac_locals.get('step') # type: int
        writer = sac_locals.get('writer') # type: tf.summary.FileWriter
        done = sac_locals.get('done', False)
        obs = sac_locals.get('obs')

        achieved_goal = obs[3:6]
        original_rew = -fetch_env_goal_distance(achieved_goal, raw_env.goal)
        original_cumulative_rew += original_rew

        if done:
            s = tf.Summary()
            s.value.add(tag='original_episode_reward', simple_value=original_cumulative_rew)
            writer.add_summary(s, step)
            original_cumulative_rew = 0.0

    model = SAC(MlpPolicy, env, verbose=1, tensorboard_log=tb_dir, **sac_params)
    model.learn(total_timesteps=steps, tb_log_name=tb_log_name, callback=callback)
    model.save(model_path)


def train_configs(config_index=None, play_only=False):

    configs = [
        dict(
            name='alpha1.0a_k2_c0.5_md0.03',
            reward_params=dict(k=2.0, c=0.5, min_dist=0.03),
            sac_params=dict(ent_coef='auto_1.0')
        ),
        dict(
            name='alpha0.1a_k2_c1_md0.03',
            reward_params=dict(k=2.0, c=1.0, min_dist=0.03),
            sac_params=dict(ent_coef='auto_0.1')
        ),
        dict(
            name='alpha1.0a_k1.5_c0.5_md0.03',
            reward_params=dict(k=1.5, c=0.5, min_dist=0.03),
            sac_params=dict(ent_coef='auto_1.0')
        ),
        dict(
            name='alpha0.1a_k1.5_c1_md0.03',
            reward_params=dict(k=1.5, c=1.0, min_dist=0.03),
            sac_params=dict(ent_coef='auto_0.1')
        ),
    ]

    if config_index is not None:
        configs = [configs[config_index]]

    for conf in configs:
        steps = 5_000_000
        model_path = f'{OUT_DIR}/fetch_sac_model_{conf["name"]}_s5M.pkl'
        tb_name = f'SAC_{conf["name"]}'
        seed = 42
        if play_only:
            play(model_path, conf['reward_params'])
        else:
            train(conf['reward_params'], conf['sac_params'], steps, model_path, seed, tb_log_name=tb_name)


def play(model_path, reward_params=None):
    env = DummyVecEnv([init_env(seed=42, reward_params=reward_params)])
    model = SAC.load(model_path)
    while True:
        obs = env.reset()
        for _ in range(100):
            env.render()
            u = model.predict(obs, deterministic=True)[0]
            obs, rew, done, info = env.step(u)
            print(rew)


def main():

    reward_params = dict(k=2.0, c=0.5, min_dist=0.03)
    model_path = f'{OUT_DIR}/model_sac_mlp_fetch_500k.pkl'

    if TRAIN:
        train(reward_params, dict(), 500_000, model_path)
    else:
        play(model_path, reward_params)


if __name__ == '__main__':
    # main()

    # should be able to do 1M steps with 4 workers in approx 2h
    train_configs(3, play_only=True)
