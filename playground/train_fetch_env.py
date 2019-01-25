import os

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C
import gym
from gym.wrappers import FlattenDictWrapper


TRAIN = True


def init_env(seed=0):

    def _init():
        env = gym.make('FetchPickAndPlaceDense-v1')
        # env = gym.make('CartPole-v1')
        env.seed(seed)
        env = FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

        # env = Monitor(
        #     env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
        #     info_keywords=('is_success',), allow_early_resets=allow_early_resets)
        # env.seed(seed)
        return env

    # env.reset()
    return _init


def main():

    tb_dir = f'./tensorboard/'
    os.makedirs(tb_dir, exist_ok=True)

    env = SubprocVecEnv([init_env(seed=42+i) for i in range(6)])
    # env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

    if TRAIN:
        model = A2C(MlpPolicy, env, verbose=1, tensorboard_log=tb_dir)
        model.learn(total_timesteps=5_000_000)
        model.save('./test_model_vec5M.pkl')
    else:
        model = A2C.load('./test_model_vec5M.pkl')

    while True:
        obs = env.reset()
        for _ in range(100):
            env.render()
            u = model.predict(obs)[0]
            obs, rew, done, info = env.step(u)


if __name__ == '__main__':
    main()
