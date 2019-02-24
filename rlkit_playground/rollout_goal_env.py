import pickle
import glob

import _rlkit
from rlkit.core import logger
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch import pytorch_util as ptu
from rlkit.launchers.config import LOCAL_LOG_DIR


def simulate_policy(file_path, gpu=False, max_path_length=50):

    data = pickle.load(open(file_path, "rb"))
    policy = data['policy']
    env = data['env']
    print("Policy and environment loaded")

    if gpu:
        ptu.set_gpu_mode(True)
        policy.to(ptu.device)

    if hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env.enable_render()

    policy.train(False)
    paths = []
    while True:
        paths.append(multitask_rollout(
            env,
            policy,
            max_path_length=max_path_length,
            animated=True,
            observation_key='observation',
            desired_goal_key='desired_goal',
        ))
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        if hasattr(env, "get_diagnostics"):
            for k, v in env.get_diagnostics(paths).items():
                logger.record_tabular(k, v)
        logger.dump_tabular()


if __name__ == '__main__':
    params_path = glob.glob(LOCAL_LOG_DIR + '/mordor/her-tsac*/her*/params.pkl')[0]
    simulate_policy(params_path)
