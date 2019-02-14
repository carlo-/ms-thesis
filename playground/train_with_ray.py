import os

import ray
import ray.tune as tune
import ray.rllib.agents.ddpg as ddpg
from ray.tune.registry import register_env


OUT_DIR = os.path.join(os.path.dirname(__file__), '../out/ray')
os.makedirs(OUT_DIR, exist_ok=True)
print(OUT_DIR)


def fetch_env_creator(env_config):
    import gym
    from gym.wrappers import FlattenDictWrapper
    env = gym.make("FetchPickAndPlaceDense-v1")
    raw_env = env.unwrapped
    env_config = env_config or dict()
    for k, v in env_config.items():
        setattr(raw_env, k, v)
    env = FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
    return env


custom_fetch_env_id = "custom_fetch"
register_env(custom_fetch_env_id, fetch_env_creator)


def get_ddpg_config(env_id, env_config=None):
    config = ddpg.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 1
    config["num_workers"] = 6
    config["policy_delay"] = 2
    config["actor_hiddens"] = [256, 256, 256]
    config["critic_hiddens"] = [256, 256, 256]
    config["buffer_size"] = 1_000_000
    config["env"] = env_id
    if isinstance(env_config, dict):
        config['env_config'] = env_config
    return config


def get_td3_config(env_id, env_config=None):
    config = get_ddpg_config(env_id, env_config)
    config["twin_q"] = True
    config["smooth_target_policy"] = True
    return config


def main():
    ray.init()

    tune.run_experiments([
        # tune.Experiment(
        #     name='fetch_ddpg_td3',
        #     run='DDPG',
        #     stop=dict(training_iteration=1_000_000),
        #     local_dir=OUT_DIR,
        #     checkpoint_freq=10,
        #     config=get_td3_config("FetchPickAndPlace-v1"),
        # ),

        # tune.Experiment(
        #     name='fetch_huber_ddpg',
        #     run='DDPG',
        #     stop=dict(training_iteration=1_000_000),
        #     local_dir=OUT_DIR,
        #     checkpoint_freq=10,
        #     config=get_ddpg_config(
        #         custom_fetch_env_id,
        #         env_config=dict(
        #             reward_params=dict(huber_loss=True)
        #         )
        #     ),
        # ),

        tune.Experiment(
            name='fetch_huber_ppo_gae',
            run='PPO',
            stop=dict(training_iteration=1_000_000),
            local_dir=OUT_DIR,
            checkpoint_freq=10,
            config={
                'env': custom_fetch_env_id,
                'env_config': dict(
                    reward_params=dict(huber_loss=True)
                ),
                'gamma': 0.995,
                'lambda': 0.95,
                'clip_param': 0.2,
                'kl_coeff': 1.0,
                'num_sgd_iter': 20,
                'lr': .0001,
                'sgd_minibatch_size': 3276,
                'horizon': 5000,
                'train_batch_size': 32000,
                'model': {'free_log_std': True},
                'num_workers': 6,
                'num_gpus': 1,
                'batch_mode': 'complete_episodes'
            },
        )

        # tune.Experiment(
        #     name='humanoid_ppo_gae',
        #     run='PPO',
        #     stop=dict(episode_reward_mean=6000),
        #     local_dir=OUT_DIR,
        #     checkpoint_freq=10,
        #     config={
        #         'env': 'Humanoid-v2',
        #         'gamma': 0.995,
        #         'lambda': 0.95,
        #         'clip_param': 0.2,
        #         'kl_coeff': 1.0,
        #         'num_sgd_iter': 20,
        #         'lr': .0001,
        #         'sgd_minibatch_size': 3276,
        #         'horizon': 5000,
        #         'train_batch_size': 32000,
        #         'model': {'free_log_std': True},
        #         'num_workers': 6,
        #         'num_gpus': 1,
        #         'batch_mode': 'complete_episodes'
        #     },
        # )
    ])


if __name__ == '__main__':
    main()
