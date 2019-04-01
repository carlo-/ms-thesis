import os
import glob
import tempfile
from datetime import datetime

import numpy as np
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.gail import ExpertDataset
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines import PPO2, SAC, DDPG
from stable_baselines.sac.policies import MlpPolicy as MlpPolicySAC
from stable_baselines.ddpg.policies import MlpPolicy as MlpPolicyDDPG

import gym
from gym.wrappers import FlattenDictWrapper
from gym.envs.robotics.fetch_env import goal_distance as fetch_env_goal_distance


ALG_NAME = 'ppo2'
OUT_DIR = f'../out/{ALG_NAME}'
os.makedirs(OUT_DIR, exist_ok=True)
current_epoch = None


def init_env(*, env_id, seed=0, env_kwargs=None):
    def _init():
        kwargs = env_kwargs or dict()
        env = gym.make(env_id, **kwargs)
        env.seed(seed)
        if isinstance(env.unwrapped, gym.GoalEnv):
            env = FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
        env = Monitor(env, None, allow_early_resets=True)
        return env
    return _init


def unnormalize_obs(obs: np.ndarray, env: VecNormalize):
    return obs * np.sqrt(env.obs_rms.var + env.epsilon) + env.obs_rms.mean


def normalize_demo(expert_path: str, env: VecNormalize):
    env.training = True
    data = np.load(expert_path)
    data = {k: data[k] for k in data.files}

    if env.norm_obs:
        obs = data['obs']
        env.obs_rms.update(obs)
        data['obs'] = np.clip((obs-env.obs_rms.mean)/np.sqrt(env.obs_rms.var+env.epsilon), -env.clip_obs, env.clip_obs)

    if env.norm_reward:
        rews = data['rewards']
        returns = data['episode_returns']
        env.ret_rms.update(rews)
        data['rewards'] = np.clip(rews / np.sqrt(env.ret_rms.var + env.epsilon), -env.clip_reward, env.clip_reward)
        data['episode_returns'] = np.clip(returns / np.sqrt(env.ret_rms.var + env.epsilon), -env.clip_reward, env.clip_reward)

    tempdir = tempfile.mkdtemp()
    normalized_file = f'{tempdir}/data.npz'
    np.savez(normalized_file, **data)
    return normalized_file


def train(*, env_id, env_kwargs, ppo_params, epochs, training_steps=0, expert_path, seed=42, n_cpus=1):

    if n_cpus == 1:
        env = DummyVecEnv([init_env(env_id=env_id, seed=seed, env_kwargs=env_kwargs)])
    else:
        env = SubprocVecEnv([init_env(env_id=env_id, seed=seed+i, env_kwargs=env_kwargs) for i in range(n_cpus)])

    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=200.)
    model = PPO2(MlpPolicy, env, verbose=1, **ppo_params)
    # model = DDPG(MlpPolicyDDPG, env, verbose=1)
    # model = SAC(MlpPolicySAC, env, verbose=1, learning_starts=0, ent_coef=0.0, **ppo_params)

    normalized_expert_path = normalize_demo(expert_path, env)

    dataset = ExpertDataset(expert_path=normalized_expert_path, batch_size=128)
    model.pretrain(dataset, n_epochs=epochs)
    if training_steps > 0:
        model.learn(total_timesteps=training_steps, seed=seed)
    env.training = False

    obs = env.reset()
    reward_sum = 0.0
    for _ in range(100_000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        reward_sum += reward
        env.render(mode='rgb_array', rgb_options=dict(camera_id=-1))
        if np.any(done):
            print(reward_sum)
            reward_sum = 0.0
            obs = env.reset()


if __name__ == '__main__':

    train(
        env_id='HandPickAndPlace-v0',
        env_kwargs=dict(
            ignore_rotation_ctrl=True,
            ignore_target_rotation=True,
            randomize_initial_arm_pos=True,
            randomize_initial_object_pos=True,
            distance_threshold=0.05,
            object_id='teapot',
            reward_type='sparse',
            object_cage=True
        ),
        ppo_params=dict(),
        n_cpus=7,
        training_steps=100_000,
        epochs=300,
        expert_path='../demonstrations/stable_baselines/hand_demo_500_teapot.npz',
    )
