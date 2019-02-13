import os

import ray
import ray.tune as tune
import ray.rllib.agents.ddpg as ddpg
import ray.rllib


EXAMPLES_DIR = ray.rllib
OUT_DIR = os.path.join(os.path.dirname(__file__), '../out/ray')
os.makedirs(OUT_DIR, exist_ok=True)
print(OUT_DIR)


def get_td3_config(env_id):
    config = ddpg.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 2
    config["twin_q"] = True

    config["smooth_target_policy"] = True
    config["policy_delay"] = 2

    config["actor_hiddens"] = [256, 256, 256]
    config["critic_hiddens"] = [256, 256, 256]

    config["buffer_size"] = 1_000_000
    config["env"] = env_id
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
        tune.Experiment(
            name='humanoid_ppo_gae',
            run='PPO',
            stop=dict(episode_reward_mean=6000),
            local_dir=OUT_DIR,
            checkpoint_freq=10,
            config={
                'env': 'Humanoid-v2',
                'gamma': 0.995,
                'lambda': 0.95,
                'clip_param': 0.2,
                'kl_coeff': 1.0,
                'num_sgd_iter': 20,
                'lr': .0001,
                'sgd_minibatch_size': 32768,
                'horizon': 5000,
                'train_batch_size': 320000,
                'model': {'free_log_std': True},
                'num_workers': 1,
                'num_gpus': 0,
                'batch_mode': 'complete_episodes'
            },
        )
    ])


if __name__ == '__main__':
    main()
