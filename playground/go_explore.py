from collections import OrderedDict
import copy
from time import sleep

import numpy as np
import gym
from gym.envs.robotics import FetchPickAndPlaceEnv
from tqdm import tqdm

POS_BINS = np.linspace(0.0, 2.0, 15)
GRIPPER_BINS = np.linspace(0.0, 0.15, 4)


def init_env(seed=0):
    env = gym.make('FetchPickAndPlace-v1')
    env.seed(seed)
    return env


def get_state(raw_env: FetchPickAndPlaceEnv):
    sim_state = copy.deepcopy(raw_env.sim.get_state())
    goal = raw_env.goal.copy()
    return sim_state, goal


def set_state(raw_env: FetchPickAndPlaceEnv, sim_state, goal):
    raw_env.sim.set_state(copy.deepcopy(sim_state))
    raw_env.sim.forward()
    raw_env.goal = goal.copy()


def bin_obs(obs):
    if not isinstance(obs, np.ndarray):
        obs = obs['observation']
    gripper_pos = obs[:3]
    object_pos = obs[3:6]
    gripper_state = obs[9:11].sum()

    dig1 = np.digitize(np.r_[gripper_pos, object_pos], POS_BINS, right=False)
    dig2 = np.digitize(gripper_state, GRIPPER_BINS, right=False)
    return tuple(np.r_[dig1, dig2])


class Cell:
    def __init__(self, *, trajectory=None, sim_state=None, goal=None, raw_env=None,
                 root_cell=None, obs=None, reward=None):

        if obs is None or reward is None:
            raise ValueError
        self.obs = obs
        self.binned_obs = bin_obs(obs)
        self.reward = reward

        self.root_cell = root_cell
        if trajectory is None:
            assert root_cell is None
            self.trajectory = []
        else:
            self.trajectory = trajectory

        if raw_env is None:
            assert sim_state is not None and goal is not None
            sim_state = copy.deepcopy(sim_state)
            goal = goal.copy()
        else:
            assert sim_state is None and goal is None
            sim_state, goal = get_state(raw_env)

        self.sim_state = sim_state
        self.goal = goal
        self.n_chosen = 0
        self.n_visited_in_expl = 0
        self.n_children_updated = 0

    def apply(self, raw_env):
        set_state(raw_env, self.sim_state, self.goal)

    @property
    def weight(self):
        a = 1. / (self.n_chosen + 0.001)
        b = 1. / (self.n_visited_in_expl + 0.001)
        c = 1. / (self.n_children_updated + 0.001)
        return a + b + c + 1.0


def phase1():

    render = False

    env = init_env(seed=42)
    raw_env = env.unwrapped # type: FetchPickAndPlaceEnv
    raw_env.reward_type = 'dense'
    action_space = env.action_space
    all_archives = []

    for r in range(10):

        obs = env.reset()
        init_reward = raw_env.compute_reward(obs['achieved_goal'], raw_env.goal, {})
        root_cell = Cell(raw_env=raw_env, obs=obs, reward=init_reward)

        archive, weights = OrderedDict(), OrderedDict()
        all_archives.append(archive)

        archive[root_cell.binned_obs] = root_cell
        weights[root_cell.binned_obs] = root_cell.weight

        for _ in tqdm(range(5_000)):
            probs = np.asarray(list(weights.values()))
            probs /= probs.sum()

            sel_cell_i = np.random.choice(len(archive), p=probs)
            sel_cell = list(archive.values())[sel_cell_i]
            sel_cell.n_chosen += 1
            sel_cell.apply(raw_env)
            if render:
                env.render()

            # root_cell = cell.root_cell or cell
            traj = list(sel_cell.trajectory)

            for _ in range(50):
                a = action_space.sample()
                obs, reward, _, _ = env.step(a)
                traj.append(a)

                if render:
                    env.render()

                add_to_archive = True
                binned = bin_obs(obs)

                if binned in archive:
                    # print('Already in archive!')

                    existing_cell = archive[binned]
                    existing_cell.n_visited_in_expl += 1
                    weights[binned] = existing_cell.weight

                    shorter_traj = len(traj) < len(existing_cell.trajectory)
                    higher_rew = reward > existing_cell.reward
                    add_to_archive = shorter_traj or higher_rew
                    # if add_to_archive:
                    #     print('Better!')

                if add_to_archive:
                    sel_cell.n_children_updated += 1
                    new_cell = Cell(raw_env=raw_env, root_cell=root_cell, trajectory=list(traj), obs=obs, reward=reward)
                    new_cell.n_visited_in_expl += 1
                    archive[binned] = new_cell
                    weights[binned] = new_cell.weight

            weights[sel_cell.binned_obs] = sel_cell.weight

        best_cell = None
        best_reward = -np.inf
        for cell in archive.values():
            if cell.reward > best_reward:
                best_cell = cell
                best_reward = cell.reward
        best_traj = best_cell.trajectory

        while True:
            print('Visualizing best trajectory...')
            root_cell.apply(raw_env)
            env.render()
            for a in best_traj:
                env.step(a)
                env.render()
                # sleep(1/60.)


if __name__ == '__main__':
    phase1()
