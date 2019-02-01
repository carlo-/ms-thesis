import argparse
import os

from mpi4py import MPI

import stable_baselines.common.tf_util as tf_util
from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy

import gym
from gym.wrappers import FlattenDictWrapper
from gym.envs.robotics import FetchEnv


def make_env(seed, rank, monitor=True, reward_params=None):
    set_global_seeds(seed)

    env = gym.make('FetchPickAndPlaceDense-v1')
    raw_env = env.unwrapped # type: FetchEnv
    raw_env.reward_params = reward_params or dict(k=1.0, c=1.0)

    env = FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
    # env = Monitor(env, os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
    if monitor:
        env = Monitor(
            env, os.path.join(logger.get_dir(), str(rank)),
            info_keywords=('is_success',), allow_early_resets=True
        )
    env.seed(seed)
    env = DummyVecEnv([lambda: env])
    return env


def train(num_timesteps, seed, model_path, tensorboard_log, reward_params=None):

    with tf_util.single_threaded_session():

        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            logger.configure()
        else:
            tensorboard_log = None
            logger.configure(format_strs=[])
            logger.set_level(logger.DISABLED)

        worker_seed = seed + 10000 * rank
        env = make_env(worker_seed, rank, monitor=False, reward_params=reward_params)

        policy_kwargs = dict()

        model = TRPO(MlpPolicy, env, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                     entcoeff=0.0, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3, tensorboard_log=tensorboard_log,
                     policy_kwargs=policy_kwargs)
        model.learn(total_timesteps=num_timesteps)
        env.close()

        model.save(f'{model_path}_{rank}.pkl')


def play(model_path):

    model = TRPO.load(model_path)
    env = make_env(42, 0, monitor=False)

    while True:
        obs = env.reset()
        for _ in range(100):
            env.render()
            u = model.predict(obs, deterministic=True)[0]
            obs, rew, done, info = env.step(u)
            print(rew)


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--play', type=bool, default=False)
    parser.add_argument('--num-timesteps', type=int, default=int(5e6))
    parser.add_argument('--model-path', type=str, default='./model')
    parser.add_argument('--tb-path', type=str, default='./tensorboard')

    args = parser.parse_args()
    if args.play:
        play(model_path=args.model_path)
    else:
        train(num_timesteps=args.num_timesteps, seed=args.seed,
              model_path=args.model_path, tensorboard_log=args.tb_path)


if __name__ == '__main__':
    # main()
    # train(num_timesteps=5_000_000, seed=42, model_path='../out/model_trpo2.pkl', tensorboard_log='../out/tensorboard2',
    #       reward_params=dict(k=2.0, c=1.0))
    play('../out/model_trpo2.pkl_0.pkl')
