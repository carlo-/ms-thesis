import os
from time import sleep

import numpy as np
import gym
import ray
import ray.tune as tune
import ray.rllib.agents.ddpg as ddpg
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.tune.registry import register_env


OUT_DIR = os.path.join(os.path.dirname(__file__), '../out/ray')
os.makedirs(OUT_DIR, exist_ok=True)
print(OUT_DIR)


def fetch_is_success(obs, has_object=True, distance_threshold=0.05):
    if has_object:
        achieved = obs[3:6]
    else:
        achieved = obs[:3]
    desired = obs[-3:]
    d = np.linalg.norm(achieved - desired, axis=-1)
    return (d < distance_threshold).astype(np.float32)


def fetch_on_episode_end(info):
    episode = info["episode"] # type: MultiAgentEpisode
    obs = episode.last_observation_for()
    success = fetch_is_success(obs)
    episode.custom_metrics["success"] = success


def get_fetch_env_creator(env_id):
    def fetch_env_creator(env_config):
        from gym.wrappers import FlattenDictWrapper
        env = gym.make(env_id)
        raw_env = env.unwrapped
        env_config = env_config or dict()
        for k, v in env_config.items():
            setattr(raw_env, k, v)
        env = FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
        return env
    return fetch_env_creator


def register_bullet_envs():
    import pybulletgym.envs
    bullet_envs = filter(lambda x: "Bullet" in x, gym.envs.registration.registry.env_specs)
    for env_id in bullet_envs:
        def make(*args):
            # this will be called by a different process, so we need to import
            # to make sure that the PyBullet envs are registered in that context
            import gym
            import pybulletgym.envs
            return gym.make(env_id)
        register_env(env_id, make)


custom_fetch_env_id = "custom_fetch"
custom_fetch_sparse_env_id = "custom_fetch_sparse"
custom_simple_fetch_env_id = "simple_custom_fetch"
register_env(custom_fetch_env_id, get_fetch_env_creator("FetchPickAndPlaceDense-v1"))
register_env(custom_fetch_sparse_env_id, get_fetch_env_creator("FetchPickAndPlace-v1"))
register_env(custom_simple_fetch_env_id, get_fetch_env_creator("FetchPickAndPlaceEasyDense-v1"))
register_bullet_envs()


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


def rollout(*, agent_type=None, agent_config=None, checkpoint_path=None, experiment=None,
            agent=None, env_id=None, num_steps):

    if experiment is not None:
        import glob
        import os

        agent_type = experiment.spec["run"]
        agent_config = experiment.spec["config"]
        env_id = agent_config['env']

        if checkpoint_path is None:
            exp_dir = f'{experiment.spec["local_dir"]}/{experiment.name}'
            checkpoints = glob.glob(f'{exp_dir}/{agent_type}_{env_id}_*/checkpoint_*/checkpoint-*[0-9]')
            assert len(checkpoints) > 0, "No checkpoints found!"
            checkpoint_path = sorted(checkpoints, key=os.path.getmtime)[-1]

    if agent is None:
        from ray.rllib.agents.registry import get_agent_class
        cls = get_agent_class(agent_type)
        agent_config['num_workers'] = 1
        agent = cls(env=env_id, config=agent_config)
        agent.restore(checkpoint_path)
        print('Restored checkpoint at:', checkpoint_path)

    if hasattr(agent, "local_evaluator"):
        env = agent.local_evaluator.env
        state_init = agent.local_evaluator.policy_map["default"].get_initial_state()
    else:
        env = gym.make(env_id)
        state_init = []

    if state_init:
        use_lstm = True
    else:
        use_lstm = False

    if 'Bullet' in env_id:
        env.render(mode='human')

    steps = 0
    while steps < (num_steps or steps + 1):
        state = env.reset()
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            if use_lstm:
                action, state_init, logits = agent.compute_action(
                    state, state=state_init)
            else:
                action = agent.compute_action(state)
            next_state, reward, done, _ = env.step(action)
            reward_total += reward
            env.render(mode='human')
            steps += 1
            state = next_state
            sleep(1/60)
        print("Episode reward", reward_total)


def main(rollout_only=False):
    ray.init()

    experiments = [
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

        # tune.Experiment(
        #     name='fetch_huber_ppo_gae',
        #     run='PPO',
        #     stop=dict(time_total_s=3600*8, training_iteration=10_000),
        #     local_dir=OUT_DIR,
        #     checkpoint_freq=10,
        #     config={
        #         'env': custom_fetch_env_id,
        #         'env_config': dict(
        #             reward_params=dict(huber_loss=True)
        #         ),
        #         'gamma': 0.995,
        #         'lambda': 0.95,
        #         'clip_param': 0.2,
        #         'kl_coeff': 1.0,
        #         'num_sgd_iter': 20,
        #         'lr': .0001,
        #         'sgd_minibatch_size': 32768,
        #         'horizon': 5000,
        #         'train_batch_size': 320000,
        #         'model': {'free_log_std': True},
        #         'num_workers': 20,
        #         'num_gpus': 4,
        #         'batch_mode': 'complete_episodes'
        #     },
        # ),

        # tune.Experiment(
        #     name='fetch_simple_huber_ppo_gae',
        #     run='PPO',
        #     stop=dict(time_total_s=3600*8, training_iteration=10_000),
        #     local_dir=OUT_DIR,
        #     checkpoint_freq=10,
        #     config={
        #         'env': custom_simple_fetch_env_id,
        #         'env_config': dict(
        #             reward_params=dict(huber_loss=True, c1=0.0)
        #         ),
        #         'callbacks': {
        #             'on_episode_end': tune.function(fetch_on_episode_end),
        #         },
        #         'gamma': 0.995,
        #         'lambda': 0.95,
        #         'clip_param': 0.2,
        #         'kl_coeff': 1.0,
        #         'num_sgd_iter': 20,
        #         'lr': .0001,
        #         'sgd_minibatch_size': 32768,
        #         'horizon': 5000,
        #         'train_batch_size': 320000,
        #         'model': {'free_log_std': True},
        #         'num_workers': 20,
        #         'num_gpus': 4,
        #         'batch_mode': 'complete_episodes'
        #     },
        # ),

        tune.Experiment(
            name='fetch_sparse_apex_ddpg',
            run='APEX_DDPG',
            stop=dict(time_total_s=3600 * 5, training_iteration=10_000),
            local_dir=OUT_DIR,
            checkpoint_freq=10,
            config={
                'env': custom_fetch_sparse_env_id,
                'callbacks': {
                    'on_episode_end': tune.function(fetch_on_episode_end),
                },
                'clip_rewards': False,
                'num_workers': 16,
                'noise_scale': 1.0,
                'n_step': 3,
                'target_network_update_freq': 50000,
                'tau': 0.001,
                'actor_hiddens': [256, 256, 256],
                'critic_hiddens': [256, 256, 256],
                'buffer_size': 1_000_000,
            },
        ),

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
        # ),

        # tune.Experiment(
        #     name='bullet_humanoid_ppo_gae',
        #     run='PPO',
        #     stop=dict(time_total_s=3600*8, training_iteration=10_000),
        #     local_dir=OUT_DIR,
        #     checkpoint_freq=100,
        #     config={
        #         'env': 'HumanoidPyBulletEnv-v0',
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
        #         'num_workers': 20,
        #         'num_gpus': 1,
        #         'batch_mode': 'complete_episodes'
        #     },
        # ),
    ]

    if rollout_only:
        for e in experiments:
            rollout(experiment=e, num_steps=10_000)
    else:
        tune.run_experiments(experiments)


if __name__ == '__main__':
    main()
