import copy

import numpy as np
import gym
from gym.envs.robotics import FetchPickAndPlaceEnv


POS_BINS = np.linspace(0.0, 2.0, 20)
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
    def __init__(self, *, trajectory=None, sim_state=None, goal=None, raw_env=None, root_cell=None, obs=None):

        if obs is None:
            raise ValueError
        self.obs = obs
        self.binned_obs = bin_obs(obs)

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

    def apply(self, raw_env):
        set_state(raw_env, self.sim_state, self.goal)

    @property
    def score(self):
        return 1. / (self.n_chosen + 0.001)


def phase1():

    env = init_env(seed=42)
    raw_env = env.unwrapped # type: FetchPickAndPlaceEnv
    action_space = env.action_space
    all_archives = []

    for r in range(10):

        obs = env.reset()
        archive = [Cell(raw_env=raw_env, obs=obs)]
        weights = [archive[0].score]
        all_archives.append(archive)

        for _ in range(100):
            probs = np.asarray(weights)
            probs /= probs.sum()

            cell_i = np.random.choice(len(archive), p=probs)
            cell = archive[cell_i]
            cell.n_chosen += 1
            weights[cell_i] = cell.score
            cell.apply(raw_env)
            env.render()

            root_cell = cell.root_cell or cell
            traj = list(cell.trajectory)

            for _ in range(10):
                a = action_space.sample()
                obs = env.step(a)[0]
                env.render()
                traj.append(a)

                # if interesting enough
                binned = bin_obs(obs)

                new_cell = Cell(raw_env=raw_env, root_cell=root_cell, trajectory=list(traj))
                archive.append(new_cell)
                weights.append(new_cell.score)


if __name__ == '__main__':
    phase1()
