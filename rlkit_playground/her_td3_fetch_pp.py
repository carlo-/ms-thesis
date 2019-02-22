import gym
import _rlkit
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import GaussianAndEpislonStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.her.her import HerTd3
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy

ptu.set_gpu_mode(True)


def experiment(variant):
    env = gym.make('FetchPickAndPlace-v1')
    es = GaussianAndEpislonStrategy(
        action_space=env.action_space,
        max_sigma=.2,
        min_sigma=.2,  # constant sigma
        epsilon=.3,
    )
    obs_dim = env.observation_space.spaces['observation'].low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    action_dim = env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        **variant['replay_buffer_kwargs']
    )
    algorithm = HerTd3(
        her_kwargs=dict(
            observation_key='observation',
            desired_goal_key='desired_goal'
        ),
        td3_kwargs=dict(
            env=env,
            qf1=qf1,
            qf2=qf2,
            policy=policy,
            exploration_policy=exploration_policy
        ),
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


def main():
    variant = dict(
        algo_kwargs=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=50,
            batch_size=256,
            discount=0.99,
        ),
        replay_buffer_kwargs=dict(
            max_size=1000000,
            fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0.0,
        ),
    )
    setup_logger('her_td3_fetch_pp', variant=variant)
    experiment(variant)


if __name__ == "__main__":
    main()
