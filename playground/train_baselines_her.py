import os
import pickle
import itertools as it

import gym
from baselines.bench import Monitor
from baselines import logger
from baselines.her import her
from baselines.her.ddpg import DDPG
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


TRAIN = False
SAVE_GIF = False

ENV_ID = 'FetchPickAndPlace-v1'
ENV_KWARGS = dict()
OUT_DIR = '../out/ddpg2'

LOG_DIR = f'{OUT_DIR}/log'
MODEL_DIR = f'{OUT_DIR}/models'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def init_env(seed=0):
    env = gym.make(ENV_ID, **ENV_KWARGS)
    env.seed(seed)
    env = Monitor(env,
                  logger.get_dir() and os.path.join(logger.get_dir(), '0.0'),
                  allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    return env


def train(model_dir):

    seed = 42
    total_timesteps = 2_500_000
    env = init_env(seed=seed)

    model = her.learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        network='mlp',
        save_path=model_dir
    )

    final_model_path = os.path.join(MODEL_DIR, 'policy_final.pkl')
    model.save(final_model_path)
    return model


def load_policy(model_path):
    with open(model_path, 'rb') as f:
        obj = pickle.load(f)
        if not isinstance(obj, DDPG):
            raise IOError
    return obj


def play(model_path):

    model = load_policy(model_path)
    env = init_env()
    obs = env.reset()
    images = []

    for i in it.count():
        actions = model.step(obs)[0]
        obs, _, done, _ = env.step(actions)
        if SAVE_GIF:
            img = env.render(mode='rgb_array')
            images.append(img)
            if i == 240:
                break
        else:
            env.render()
        if done:
            obs = env.reset()

    if SAVE_GIF:
        import imageio
        imageio.mimwrite(f'{OUT_DIR}/out.gif', images, fps=60)


def main():

    logger.configure(dir=LOG_DIR, format_strs=('stdout', 'tensorboard'))

    if TRAIN:
        train(MODEL_DIR)

    latest_model_path = os.path.join(MODEL_DIR, 'policy_latest.pkl')
    if os.path.exists(latest_model_path):
        play(latest_model_path)


if __name__ == '__main__':
    main()

    # play(os.path.join(MODEL_DIR, 'policy_610.pkl'))
    # play(os.path.join(MODEL_DIR, 'policy_375.pkl'))
