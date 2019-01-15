from time import sleep
import itertools as it

import numpy as np
import gym


def set_body_pose(model, body_id, pose):
    model.body_pos[body_id, :] = pose[:3]
    model.body_quat[body_id, :] = pose[3:]


def get_body_pose(model, body_id):
    pos = model.body_pos[body_id]
    quat = model.body_quat[body_id]
    return np.r_[pos, quat]


def main():
    env = gym.make('HandReach-v0')
    env.reset()

    model = env.unwrapped.sim.model
    mount_id = model._body_name2id['robot0:hand mount']

    p_orig = get_body_pose(model, mount_id)

    for i in it.count():

        p = get_body_pose(model, mount_id)
        p[:3] = p_orig[:3] + np.sin(i/10)*0.05
        p[3:] = p_orig[3:] + np.sin(i/10)*0.5
        set_body_pose(model, mount_id, p)

        env.render()
        action = env.action_space.sample()
        env.step(action)
        sleep(1/60)


if __name__ == '__main__':
    main()
