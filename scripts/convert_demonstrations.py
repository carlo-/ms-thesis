import pickle

import numpy as np
import gym


def her_torch_to_sb(input_file: str, output_file: str, goal_env: gym.GoalEnv=None, reward_fn=None):

    assert input_file.endswith('.pkl')
    assert output_file.endswith('.npz')

    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    # useful for debugging:
    # del data['mb_sim_states']
    # data = {k: v[:3] for k, v in data.items()}
    # from gym.agents.shadow_hand import HandPickAndPlaceAgent
    # agent = HandPickAndPlaceAgent(env)
    # from gym.wrappers import FlattenDictWrapper
    # env = FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
    # example_numpy_dict = generate_expert_traj(agent, env)

    n_episodes, ep_len, action_size = data['mb_actions'].shape
    obs_size = data['mb_obs'].shape[-1] + data['mb_g'].shape[-1]

    if callable(reward_fn):
        assert goal_env is None
    else:
        assert reward_fn is None
        reward_fn = goal_env.compute_reward
    rewards = reward_fn(data['mb_ag'][:, 1:], data['mb_g'], info=dict())
    episode_starts = np.zeros(ep_len * n_episodes, dtype=np.bool)
    episode_starts[::ep_len] = True

    numpy_dict = {
        'actions': data['mb_actions'].reshape(-1, action_size),
        'obs': np.concatenate([data['mb_obs'][:, :-1], data['mb_g']], axis=2).reshape(-1, obs_size),
        'rewards': rewards.ravel(),
        'episode_returns': rewards.sum(axis=1),
        'episode_starts': episode_starts
    }

    np.savez(output_file, **numpy_dict)


def _generate_expert_traj(model, env=None, n_episodes=3):
    # From stable_baselines.gail.dataset.record_expert
    # used here only for debugging

    actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    idx = 0

    while ep_idx < n_episodes:
        observations.append(obs)

        action = model.predict(env.unwrapped._get_obs())
        obs, reward, done, _ = env.step(action)

        actions.append(action)
        rewards.append(reward)
        episode_starts.append(done)
        reward_sum += reward
        idx += 1
        if done:
            obs = env.reset()
            episode_returns[ep_idx] = reward_sum
            reward_sum = 0.0
            ep_idx += 1

    observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
    actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)

    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts[:-1])

    assert len(observations) == len(actions)

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }

    return numpy_dict


if __name__ == '__main__':

    # env = gym.make(
    #     'HandPickAndPlace-v0',
    #     ignore_rotation_ctrl=True,
    #     ignore_target_rotation=True,
    #     randomize_initial_arm_pos=True,
    #     randomize_initial_object_pos=True,
    #     distance_threshold=0.05,
    #     object_id='small_sphere',
    #     reward_type='dense',
    # )
    #
    # her_torch_to_sb(
    #     input_file='../../misc/hindsight-experience-replay/demonstrations/hand_demo_500_small_sphere.pkl',
    #     output_file='../demonstrations/stable_baselines/hand_demo_dense_500_small_sphere.npz',
    #     goal_env=env.unwrapped
    # )

    # env = gym.make('YumiLift-v1', randomize_initial_object_pos=False)
    #
    # def custom_rew_fn(achieved_goal, *args, **kwargs):
    #     idx = achieved_goal > 0.01
    #     rew = np.zeros_like(achieved_goal)
    #     # rew[:] = -0.05
    #     rew[idx] = (achieved_goal[idx] - 0.01) ** 0.1
    #     return rew
    #
    # her_torch_to_sb(
    #     input_file='../../misc/hindsight-experience-replay/demonstrations/yumi_lift_700_fixed.pkl',
    #     output_file='../demonstrations/stable_baselines/yumi_lift_700_fixed_cr3.npz',
    #     # goal_env=env.unwrapped,
    #     reward_fn=custom_rew_fn,
    # )

    env = gym.make('YumiReachTwoArms-v1', reward_type='dense')

    her_torch_to_sb(
        input_file='../../misc/hindsight-experience-replay/demonstrations/yumi_reach_two_arms_700.pkl',
        output_file='../demonstrations/stable_baselines/yumi_reach_two_arms_700.npz',
        goal_env=env.unwrapped,
    )
