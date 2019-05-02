from threading import Thread
import itertools as it
import copy

import numpy as np
import gym
from gym.utils.mjviewer import add_selection_logger
from gym.utils.transformations import render_pose
from gym.agents.yumi import YumiConstrainedAgent
from gym.utils import transformations as tf

from playground.twin_vae import TwinVAE, TwinDataset, SimpleAutoencoder
from thesis_tools.utils import wait_for_key


KEYBOARD_CTRL = False


def _flatten_obs(obs_dict):
    return np.r_[obs_dict['observation'], obs_dict['desired_goal']]


def _action_thread(yumi_action, hand_action):

    def safe_int(x):
        try:
            return int(x)
        except ValueError:
            return None

    a = yumi_action
    while True:
        key = wait_for_key()
        i = safe_int(key)
        print(key)

        if i is not None:
            if i == 5:
                a[3] = 0.2
            elif i == 6:
                a[3] = -0.2
            elif i == 2:
                a[0] = -1.
            elif i == 3:
                a[0] = 1.
        elif key == 'A':
            a[1] = -0.2
        elif key == 'B':
            a[1] = 0.2
        elif key == 'C':
            a[2] = 0.2
        elif key == 'D':
            a[2] = -0.2


def _imitate_hand_with_model(model: TwinVAE, hand_norm_obs: np.ndarray, yumi_env, yumi_scaler, n=10) -> np.ndarray:

    from gym.envs.yumi.yumi_constrained import YumiConstrainedEnv

    if not isinstance(yumi_env, YumiConstrainedEnv):
        yumi_env = yumi_env.unwrapped # type: YumiConstrainedEnv

    prev_state = copy.deepcopy(yumi_env.sim.get_state())
    best_u = None
    best_sim_loss = np.inf

    yumi_viewer = yumi_env.sim_env.viewer
    yumi_env.sim_env.viewer = None

    for _ in range(n):
        u = np.random.uniform(-1, 1, size=4) * 0.1
        u[0] = np.sign(u[0])
        yumi_obs = yumi_env.step(u.copy())[0]
        yumi_obs = yumi_scaler.transform(_flatten_obs(yumi_obs)[None], copy=True)[0]
        sim_loss = model.compute_obs_sim_loss(yumi_obs, hand_norm_obs).mean()
        if sim_loss < best_sim_loss:
            best_u = u
            best_sim_loss = sim_loss
        yumi_env.sim.set_state(copy.deepcopy(prev_state))

    yumi_env.sim_env.viewer = yumi_viewer
    return best_u


def main():

    a_env = gym.make(
        'YumiConstrained-v1',
        reward_type='sparse'
    )

    b_env = gym.make(
        'HandPickAndPlace-v0',
        reward_type='sparse',
        ignore_rotation_ctrl=True,
        ignore_target_rotation=True,
        success_on_grasp_only=False,
        randomize_initial_arm_pos=True,
        randomize_initial_object_pos=True,
        object_id='box'
    )

    yumi_action = np.zeros(a_env.action_space.shape)
    hand_action = np.zeros(b_env.action_space.shape)

    if KEYBOARD_CTRL:
        p = Thread(target=_action_thread, args=(yumi_action, hand_action))
        p.start()

    dataset = TwinDataset.load('../out/pp_yumi_twin_dataset_3k.pkl')
    dataset.normalize()

    from gym.agents.shadow_hand import HandPickAndPlaceAgent
    hand_agent = HandPickAndPlaceAgent(b_env)

    model = TwinVAE.load('../out/pp_yumi_twin_ae_test_z15/checkpoints/model_c49.pt',
                         net_class=SimpleAutoencoder)
    # model.to('cuda')

    done = True
    a_obs = b_obs = None

    for i in it.count():

        if done:
            a_env.reset()
            b_env.reset()

            a_table_tf = a_env.unwrapped.get_table_surface_pose()
            b_table_tf = b_env.unwrapped.get_table_surface_pose()

            t_to_goal = tf.get_tf(np.r_[a_env.unwrapped.goal, 1., 0., 0., 0.], a_table_tf)
            b_goal_pose = tf.apply_tf(t_to_goal, b_table_tf)

            b_env.unwrapped.goal = np.r_[b_goal_pose[:3], np.zeros(4)]

            t_to_obj = tf.get_tf(a_env.unwrapped.get_object_pose(), a_table_tf)
            b_obj_pose = tf.apply_tf(t_to_obj, b_table_tf)

            object_pos = b_env.unwrapped.sim.data.get_joint_qpos('object:joint').copy()
            object_pos[:2] = b_obj_pose[:2]
            b_env.unwrapped.sim.data.set_joint_qpos('object:joint', object_pos)
            b_env.unwrapped.sim.forward()

            a_obs = a_env.unwrapped._get_obs()
            b_obs = b_env.unwrapped._get_obs()

        b_obs_orig = b_obs
        a_obs = dataset.a_scaler.transform(_flatten_obs(a_obs)[None], copy=True)[0]
        b_obs = dataset.b_scaler.transform(_flatten_obs(b_obs)[None], copy=True)[0]

        sim_loss = model.compute_obs_sim_loss(a_obs, b_obs)
        print(sim_loss.mean())

        a_env.render()
        b_env.render()

        if not KEYBOARD_CTRL:
            yumi_action = _imitate_hand_with_model(model, b_obs, a_env, dataset.a_scaler, n=5)

        if KEYBOARD_CTRL or i % 5 == 0:
            hand_action = hand_agent.predict(b_obs_orig)
            b_obs = b_env.step(hand_action)[0]
        else:
            b_obs = b_obs_orig

        a_obs = a_env.step(yumi_action)[0]
        done = False

        if KEYBOARD_CTRL and i % 5 == 0:
            yumi_action[1:] = 0.


if __name__ == '__main__':
    main()
