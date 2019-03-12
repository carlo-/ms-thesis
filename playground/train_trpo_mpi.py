import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from mpi4py import MPI

import stable_baselines.common.tf_util as tf_util
from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy

import gym
from gym.wrappers import FlattenDictWrapper
from gym.envs.robotics.fetch_env import FetchEnv, goal_distance as fetch_env_goal_distance


ALG_NAME = 'sb_trpo'
OUT_DIR = f'../out/{ALG_NAME}'
REMOTE_OUT_DIR = f'/run/user/1000/gvfs/sftp:host=mordor.csc.kth.se,port=2222,user=carlora/home/carlora/thesis/repo/out/{ALG_NAME}'
os.makedirs(OUT_DIR, exist_ok=True)
current_epoch = None


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


def train(*, env_id, reward_params, steps, local_dir, seed=42, checkpoint_freq=1):

    with tf_util.single_threaded_session():

        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        local_dir = f'{local_dir}/{now}'
        checkpoints_dir = f'{local_dir}/checkpoints'
        normalizer_dir = f'{local_dir}/normalizer'
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(normalizer_dir, exist_ok=True)

        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            tb_log_dir = local_dir
            logger.configure()
        else:
            tb_log_dir = None
            logger.configure(format_strs=[])
            logger.set_level(logger.DISABLED)

        worker_seed = seed + 10000 * rank
        set_global_seeds(worker_seed)
        env = DummyVecEnv([init_env(env_id=env_id, seed=worker_seed, reward_params=reward_params)])
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=200.)
        model = TRPO(MlpPolicy, env, tensorboard_log=tb_log_dir, cg_damping=0.1, vf_iters=5, vf_stepsize=1e-3)

        global current_epoch
        current_epoch = -1

        def save_checkpoint(epoch):
            print('Saved checkpoint for epoch', epoch)
            model_path = f'{checkpoints_dir}/model_{epoch}.pkl'
            model.save(model_path)
            env.save_running_average(normalizer_dir)

        def callback(model_locals, model_globals):

            writer = model_locals.get('writer')  # type: tf.summary.FileWriter
            update = model_locals.get('iters_so_far')  # type: int
            step = model_locals.get('timesteps_so_far')  # type: int

            global current_epoch
            if current_epoch != update:
                current_epoch = update

                if current_epoch == 1 or (current_epoch % checkpoint_freq) == 0:
                    save_checkpoint(current_epoch)

                obs = model_locals.get('observation')
                if obs is not None and len(obs) > 0:
                    obs = unnormalize_obs(obs, env)
                    achieved_goals, goals = obs[..., 3:6], obs[..., -3:]
                    original_rewards = -fetch_env_goal_distance(achieved_goals, goals)
                    avg_original_success_rate = np.mean(-original_rewards < 0.05)
                    avg_original_rew = np.mean(original_rewards)

                    s = tf.Summary()
                    s.value.add(tag='custom/original_episode_reward', simple_value=avg_original_rew)
                    s.value.add(tag='custom/original_success_rate', simple_value=avg_original_success_rate)
                    writer.add_summary(s, step)

        if rank != 0:
            callback = None
        model.learn(total_timesteps=steps, callback=callback, tb_log_name='tb')


def play(*, env_id, run_dir, reward_params=None, epoch):

    model_path = f'{run_dir}/checkpoints/model_{epoch}.pkl'
    normalizer_dir = f'{run_dir}/normalizer'

    env = DummyVecEnv([init_env(env_id=env_id, seed=42, reward_params=reward_params)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=200., training=False)
    env.load_running_average(normalizer_dir)

    model = TRPO.load(model_path, env=env)
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
        steps=30_000_000,
        local_dir=f'{OUT_DIR}/fetch_stepped_rew_v2',
        checkpoint_freq=40,
    )
