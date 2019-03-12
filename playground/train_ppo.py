import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines import PPO2

import gym
from gym.wrappers import FlattenDictWrapper
from gym.envs.robotics import FetchPickAndPlaceEnv
from gym.envs.robotics.fetch_env import goal_distance as fetch_env_goal_distance


OUT_DIR = '../out/ppo2'
os.makedirs(OUT_DIR, exist_ok=True)
current_epoch = None


def init_env(*, env_id, seed=0, reward_params=None):
    def _init():
        env = gym.make(env_id)
        raw_env = env.unwrapped # type: FetchPickAndPlaceEnv
        raw_env.reward_params = reward_params
        env.seed(seed)
        env = FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
        env = Monitor(env, None)
        return env
    return _init


def train(*, env_id, reward_params, ppo_params, steps, local_dir, seed=42, n_cpus=1, checkpoint_freq=1):

    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    local_dir = f'{local_dir}/{now}'
    checkpoints_dir = f'{local_dir}/checkpoints'
    normalizer_dir = f'{local_dir}/normalizer'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(normalizer_dir, exist_ok=True)

    env = SubprocVecEnv([init_env(env_id=env_id, seed=seed+i, reward_params=reward_params) for i in range(n_cpus)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=200.)
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=local_dir, **ppo_params)

    global current_epoch
    current_epoch = -1

    def save_checkpoint(epoch):
        print('Saved checkpoint for epoch', epoch)
        model_path = f'{checkpoints_dir}/model_{epoch}.pkl'
        model.save(model_path)
        env.save_running_average(normalizer_dir)

    def callback(model_locals, model_globals):

        writer = model_locals.get('writer') # type: tf.summary.FileWriter
        update = model_locals.get('update') # type: int
        step = update * model.n_batch

        global current_epoch
        if current_epoch != update:
            current_epoch = update

            if current_epoch == 1 or (current_epoch % checkpoint_freq) == 0:
                save_checkpoint(current_epoch)

            obs = model_locals.get('obs')
            if obs is not None and len(obs) > 0:

                achieved_goals, goals = obs[..., 3:6], obs[..., -3:]
                original_rewards = -fetch_env_goal_distance(achieved_goals, goals)
                avg_original_rew = np.mean(original_rewards)

                print(avg_original_rew)
                s = tf.Summary()
                s.value.add(tag='original_episode_reward', simple_value=avg_original_rew)
                writer.add_summary(s, step)

    model.learn(total_timesteps=steps, callback=callback, tb_log_name='tb')


def play(*, env_id, run_dir, reward_params=None, epoch):

    model_path = f'{run_dir}/checkpoints/model_{epoch}.pkl'
    normalizer_dir = f'{run_dir}/normalizer'

    env = DummyVecEnv([init_env(env_id=env_id, seed=42, reward_params=reward_params)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=200., training=False)
    env.load_running_average(normalizer_dir)

    model = PPO2.load(model_path, env=env)
    obs = env.reset()
    while True:
        env.render()
        u = model.predict(obs, deterministic=True)[0]
        obs, rew, done, info = env.step(u)
        print(rew)


if __name__ == '__main__':

    train(
        env_id='FetchPickAndPlaceDense-v1',
        reward_params=dict(stepped=True),
        ppo_params=dict(),
        steps=100_000_000,
        local_dir=f'{OUT_DIR}/fetch_stepped_rew_v2',
        n_cpus=20,
        checkpoint_freq=40,
    )

    # play(
    #     env_id='FetchPickAndPlaceDense-v1',
    #     reward_params=dict(stepped=True),
    #     run_dir=f'{OUT_DIR}/fetch_stepped_rew/mordor',
    #     epoch=3900,
    # )
