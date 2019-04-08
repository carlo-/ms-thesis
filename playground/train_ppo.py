import os
import glob
from datetime import datetime

import numpy as np
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines import PPO2

import gym
from gym.wrappers import FlattenDictWrapper
from gym.envs.robotics.fetch_env import goal_distance as fetch_env_goal_distance


ALG_NAME = 'ppo2'
OUT_DIR = f'../out/{ALG_NAME}'
REMOTE_OUT_DIR = f'/run/user/1000/gvfs/sftp:host=mordor.csc.kth.se,port=2222,user=carlora/home/carlora/thesis/repo/out/{ALG_NAME}'
os.makedirs(OUT_DIR, exist_ok=True)
current_epoch = None


def init_env(*, env_id, seed=0, env_kwargs=None, obs_keys=None):
    def _init():
        kwargs = env_kwargs or dict()
        env = gym.make(env_id, **kwargs)
        env.seed(seed)
        if isinstance(env.unwrapped, gym.GoalEnv):
            dict_keys = obs_keys or ['observation', 'desired_goal']
            env = FlattenDictWrapper(env, dict_keys=dict_keys)
        env = Monitor(env, None)
        return env
    return _init


def unnormalize_obs(obs: np.ndarray, env: VecNormalize):
    return obs * np.sqrt(env.obs_rms.var + env.epsilon) + env.obs_rms.mean


def train(*, env_id, env_kwargs, ppo_params, steps, local_dir, seed=42, n_cpus=1, checkpoint_freq=1, obs_keys=None):

    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    local_dir = f'{local_dir}/{now}'
    checkpoints_dir = f'{local_dir}/checkpoints'
    normalizer_dir = f'{local_dir}/normalizer'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(normalizer_dir, exist_ok=True)

    is_fetch = 'Fetch' in env_id
    env = SubprocVecEnv([init_env(env_id=env_id, seed=seed+i, env_kwargs=env_kwargs, obs_keys=obs_keys) for i in range(n_cpus)])
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
            if is_fetch and obs is not None and len(obs) > 0:
                obs = unnormalize_obs(obs, env)
                achieved_goals, goals = obs[..., 3:6], obs[..., -3:]
                original_rewards = -fetch_env_goal_distance(achieved_goals, goals)
                avg_original_success_rate = np.mean(-original_rewards < 0.05)
                avg_original_rew = np.mean(original_rewards)

                s = tf.Summary()
                s.value.add(tag='custom/original_episode_reward', simple_value=avg_original_rew)
                s.value.add(tag='custom/original_success_rate', simple_value=avg_original_success_rate)
                writer.add_summary(s, step)

    model.learn(total_timesteps=steps, callback=callback, tb_log_name='tb')


def play(*, env_id, run_dir, env_kwargs=None, epoch=None, obs_keys=None):

    epoch = epoch or '*'
    model_path = '{run_dir}/checkpoints/model_{epoch}.pkl'
    normalizer_dir = f'{run_dir}/normalizer'

    if epoch == '*':
        paths = glob.glob(model_path.format(run_dir=run_dir, epoch='*'))
        if len(paths) == 0:
            raise FileNotFoundError
        epoch = np.max([int(p.split('model_')[1].split('.pkl')[0]) for p in paths])

    env = DummyVecEnv([init_env(env_id=env_id, seed=42, env_kwargs=env_kwargs, obs_keys=obs_keys)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=200., training=False)
    env.load_running_average(normalizer_dir)

    model_path = model_path.format(run_dir=run_dir, epoch=epoch)
    model = PPO2.load(model_path, env=env)
    obs = env.reset()
    while True:
        env.render()
        u = model.predict(obs, deterministic=True)[0]
        obs, rew, done, info = env.step(u)
        print(rew)


if __name__ == '__main__':

    # train(
    #     env_id='YumiReachTwoArms-v0',
    #     env_kwargs=dict(reward_type='dense'),
    #     ppo_params=dict(
    #         n_steps=256
    #     ),
    #     steps=100_000_000,
    #     local_dir=f'{OUT_DIR}/yumi_reach_test4',
    #     n_cpus=15,
    #     checkpoint_freq=20,
    # )

    # play(
    #     env_id='FetchPickAndPlaceDense-v1',
    #     env_kwargs=dict(reward_params=dict(stepped=True), explicit_goal_distance=True),
    #     run_dir=glob.glob(f'{REMOTE_OUT_DIR}/fetch_stepped_rew_v3/*')[0],
    #     epoch=19520,
    # )

    # play(
    #     env_id='YumiReachLeftArm-v0',
    #     env_kwargs=dict(reward_type='dense'),
    #     run_dir=glob.glob(f'{REMOTE_OUT_DIR}/yumi_reach_test3/*')[0],
    #     epoch=None,
    # )

    # play(
    #     env_id='YumiReachTwoArms-v0',
    #     env_kwargs=dict(reward_type='dense'),
    #     run_dir=glob.glob(f'{REMOTE_OUT_DIR}/yumi_reach_test4/*')[0],
    #     epoch=None,
    # )

    train(
        env_id='HandStepped-v0',
        env_kwargs=dict(),
        ppo_params=dict(),
        steps=5_000_000,
        local_dir=f'{OUT_DIR}/hand_stepped_test1',
        n_cpus=8,
        checkpoint_freq=10,
        obs_keys=['observation']
    )

    # play(
    #     env_id='HandStepped-v0',
    #     env_kwargs=dict(render_substeps=True),
    #     run_dir=glob.glob(f'{OUT_DIR}/hand_stepped_test1/*')[0],
    #     epoch=None,
    #     obs_keys=['observation']
    # )
