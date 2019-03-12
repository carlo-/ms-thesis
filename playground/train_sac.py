import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter

from stable_baselines.bench import Monitor
from stable_baselines.sac import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import SAC

import gym
from gym.wrappers import FlattenDictWrapper
from gym.envs.robotics.fetch_env import FetchEnv, goal_distance as fetch_env_goal_distance


ALG_NAME = 'sb_sac'
OUT_DIR = f'../out/{ALG_NAME}'
REMOTE_OUT_DIR = f'/run/user/1000/gvfs/sftp:host=mordor.csc.kth.se,port=2222,user=carlora/home/carlora/thesis/repo/out/{ALG_NAME}'
os.makedirs(OUT_DIR, exist_ok=True)
tb_info = None


def init_env(*, env_id, seed=0, reward_params=None):
    def _init():
        env = gym.make(env_id)
        raw_env = env.unwrapped # type: FetchEnv
        raw_env.reward_params = reward_params
        env.seed(seed)
        env = FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
        env = Monitor(env, None)
        return env
    return _init


def unnormalize_obs(obs: np.ndarray, env: VecNormalize):
    return obs * np.sqrt(env.obs_rms.var + env.epsilon) + env.obs_rms.mean


def train(*, env_id, reward_params, sac_params, steps, local_dir, seed=42, checkpoint_freq=10_000, log_freq=1_000):

    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    local_dir = f'{local_dir}/{now}'
    checkpoints_dir = f'{local_dir}/checkpoints'
    normalizer_dir = f'{local_dir}/normalizer'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(normalizer_dir, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=local_dir)
    env = DummyVecEnv([init_env(env_id=env_id, seed=seed, reward_params=reward_params)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=200.)
    model = SAC(MlpPolicy, env, verbose=0, tensorboard_log=None, **sac_params)

    global tb_info
    tb_info = None

    def reset_tb_info():
        global tb_info
        tb_info = dict(
            successes=[],
            orig_rewards=[]
        )

    def save_checkpoint(epoch):
        print('Saved checkpoint for epoch', epoch)
        model_path = f'{checkpoints_dir}/model_{epoch}.pkl'
        model.save(model_path)
        env.save_running_average(normalizer_dir)

    def callback(model_locals, model_globals):
        global tb_info

        step = model_locals['step'] # type: int
        done = model_locals.get('done', False)
        if done:
            if tb_info is None:
                reset_tb_info()
            info = model_locals['info']
            success = info['is_success']
            obs = model_locals['new_obs']
            obs = unnormalize_obs(obs, env)
            achieved_goal, goal = obs[..., 3:6], obs[..., -3:]
            original_reward = -fetch_env_goal_distance(achieved_goal, goal)
            tb_info['successes'].append(success)
            tb_info['orig_rewards'].append(original_reward)

        if step > 0 and step % log_freq == 0:
            start_time = model_locals['start_time']
            tb_scalars = dict(
                episodes=model_locals['num_episodes'],
                network_updates=model_locals['n_updates'],
                learning_rate=model_locals['current_lr'],
                mean_reward=model_locals['mean_reward'],
                time_elapsed=int(time.time() - start_time),
                fps=(step / int(time.time() - start_time)),
                avg_original_success_rate=np.mean(tb_info['successes']),
                avg_original_rew=np.mean(tb_info['orig_rewards']),
            )
            print('\n' + '#' * 40)
            print(f'Step {step}')
            print('#' * 40)
            for k, v in tb_scalars.items():
                print(f'{k}: {v}')
                tb_writer.add_scalar(f'data/{k}', v, step)
            tb_info = None

        if step > 0 and step % checkpoint_freq == 0:
            epoch = step // checkpoint_freq
            save_checkpoint(epoch)

    model.learn(total_timesteps=steps, callback=callback, tb_log_name='tb')


def play(*, env_id, run_dir, reward_params=None, epoch):

    model_path = f'{run_dir}/checkpoints/model_{epoch}.pkl'
    normalizer_dir = f'{run_dir}/normalizer'

    env = DummyVecEnv([init_env(env_id=env_id, seed=42, reward_params=reward_params)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=200., training=False)
    env.load_running_average(normalizer_dir)

    model = SAC.load(model_path, env=env)
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
        local_dir=f'{OUT_DIR}/fetch_stepped_rew_v2',
        steps=100_000_000,
        log_freq=1_000, # in steps
        checkpoint_freq=10_000, # in steps
        sac_params=dict(
            buffer_size=int(1e6),
            batch_size=256,
            tau=1e-2,
            learning_rate=1e-3,
        )
    )
