import numpy as np
import gym
from gym.agents.yumi import YumiConstrainedAgent
from gym.agents.shadow_hand import HandPickAndPlaceAgent
from gym.agents.fetch import FetchPickAndPlaceAgent
from gym.utils import transformations as tf

import _thesis_modules
from playground.twin_vae import TwinVAE, TwinDataset, SimpleAutoencoder, VAE


def _controller(s_obs_, t_obs_, prev_s_u_, s_action_space_) -> np.ndarray:
    u = np.zeros(s_action_space_.shape)
    pos_err = t_obs_[:3] - s_obs_[:3]
    t_g_dist = np.linalg.norm(t_obs_[6:9] - t_obs_[9:12], ord=2)
    # s_g_dist = np.linalg.norm(s_obs_['observation'][6:9] - s_obs_['observation'][9:12], ord=2)
    # g_dist_err = t_g_dist - s_g_dist
    u[1:4] = pos_err * 5.0
    u[0] = np.interp(t_g_dist, [0.05, 0.30], [-1, 1]) * 1.2
    return np.clip(u, -1, 1)


def _flatten_obs(obs_dict):
    return np.r_[obs_dict['observation'], obs_dict['desired_goal']]


def yumi_to_yumi():

    teacher_env = gym.make(
        'YumiConstrained-v1',
        reward_type='sparse'
    )

    student_env = gym.make(
        'YumiConstrained-v1',
        reward_type='sparse'
    )

    teacher = YumiConstrainedAgent(teacher_env)

    done = True
    t_obs = s_obs = prev_s_u = None

    while True:

        if done:
            teacher_env.reset()
            student_env.reset()
            prev_s_u = np.zeros(student_env.action_space.shape)

            student_env.unwrapped.sim_env.goal = teacher_env.unwrapped.sim_env.goal.copy()
            object_pos = student_env.unwrapped.sim.data.get_joint_qpos('object0:joint').copy()
            object_pos[:2] = teacher_env.unwrapped.sim.data.get_joint_qpos('object0:joint')[:2].copy()
            student_env.unwrapped.sim.data.set_joint_qpos('object0:joint', object_pos)
            student_env.unwrapped.sim.forward()

            t_obs = teacher_env.unwrapped._get_obs()
            s_obs = student_env.unwrapped._get_obs()

        t_u = teacher.predict(t_obs)
        t_obs, _, done, _ = teacher_env.step(t_u)

        s_u = _controller(s_obs['observation'], t_obs['observation'], prev_s_u, student_env.action_space)
        prev_s_u = s_u.copy()
        s_obs = student_env.step(s_u)[0]

        teacher_env.render()
        student_env.render()


def hand_to_yumi():

    # model = TwinVAE.load('../out/pp_and_reach_yumi_twin_ae_test_z15/checkpoints/model_c49.pt',
    #                      net_class=SimpleAutoencoder)

    model = TwinVAE.load('../out/twin_ae_kdl_test/checkpoints/model_c12.pt',
                         net_class=SimpleAutoencoder)

    # model = TwinVAE.load('../out/twin_vae_resets_test/checkpoints/model_c5.pt',
    #                      net_class=VAE)

    dataset = TwinDataset.merge(
        TwinDataset.load('../out/pp_yumi_twin_dataset_3k.pkl'),
        TwinDataset.load('../out/pp_reach_yumi_twin_dataset_2k.pkl')
    )
    dataset.normalize()

    teacher_env = gym.make(
        'HandPickAndPlace-v0',
        ignore_rotation_ctrl=True,
        ignore_target_rotation=True,
        success_on_grasp_only=False,
        randomize_initial_arm_pos=True,
        randomize_initial_object_pos=True,
        object_id='box'
    )

    student_env = gym.make(
        'YumiConstrained-v1',
        reward_type='sparse'
    )

    teacher = HandPickAndPlaceAgent(teacher_env)

    done = True
    t_obs = s_obs = prev_s_u = None

    s_table_tf = student_env.unwrapped.get_table_surface_pose()
    t_table_tf = teacher_env.unwrapped.get_table_surface_pose()

    while True:

        if done:
            student_env.reset()
            teacher_env.reset()
            prev_s_u = np.zeros(student_env.action_space.shape)

            tf_to_goal = tf.get_tf(np.r_[student_env.unwrapped.goal, 1., 0., 0., 0.], s_table_tf)
            t_goal_pose = tf.apply_tf(tf_to_goal, t_table_tf)

            teacher_env.unwrapped.goal = np.r_[t_goal_pose[:3], np.zeros(4)]

            tf_to_obj = tf.get_tf(student_env.unwrapped.get_object_pose(), s_table_tf)
            t_obj_pose = tf.apply_tf(tf_to_obj, t_table_tf)

            object_pos = teacher_env.unwrapped.sim.data.get_joint_qpos('object:joint').copy()
            object_pos[:2] = t_obj_pose[:2]
            teacher_env.unwrapped.sim.data.set_joint_qpos('object:joint', object_pos)
            teacher_env.unwrapped.sim.forward()

            teacher.reset()

            s_obs = student_env.unwrapped._get_obs()
            t_obs = teacher_env.unwrapped._get_obs()

        t_u = teacher.predict(t_obs)
        t_obs, _, done, _ = teacher_env.step(t_u)

        b_obs = dataset.b_scaler.transform(_flatten_obs(t_obs)[None], copy=True)[0]
        recon_t_obs = model.cross_decode_b_to_a(b_obs)
        recon_t_obs = dataset.a_scaler.inverse_transform(recon_t_obs[None], copy=True)[0]

        s_u = _controller(s_obs['observation'], recon_t_obs, prev_s_u, student_env.action_space)
        prev_s_u = s_u.copy()
        s_obs = student_env.step(s_u)[0]

        teacher_env.render()
        student_env.render()


def fetch_to_yumi():

    model = TwinVAE.load('../out/twin_yumi_fetch_ae_resets/checkpoints/model_c3.pt',
                         net_class=SimpleAutoencoder)

    dataset = TwinDataset.load('../out/pp_yumi_fetch_twin_dataset_5k.pkl')
    dataset.normalize()

    teacher_env = gym.make(
        'FetchPickAndPlace-v1',
        reward_type='sparse'
    )

    student_env = gym.make(
        'YumiConstrained-v1',
        reward_type='sparse'
    )

    teacher = FetchPickAndPlaceAgent(teacher_env)

    done = True
    t_obs = s_obs = prev_s_u = None

    s_table_tf = student_env.unwrapped.get_table_surface_pose()
    t_table_tf = gym.make('HandPickAndPlace-v0').unwrapped.get_table_surface_pose()

    while True:

        if done:
            student_env.reset()
            teacher_env.reset()
            prev_s_u = np.zeros(student_env.action_space.shape)

            tf_to_goal = tf.get_tf(np.r_[student_env.unwrapped.goal, 1., 0., 0., 0.], s_table_tf)
            t_goal_pose = tf.apply_tf(tf_to_goal, t_table_tf)

            teacher_env.unwrapped.goal = t_goal_pose[:3]

            tf_to_obj = tf.get_tf(student_env.unwrapped.get_object_pose(), s_table_tf)
            t_obj_pose = tf.apply_tf(tf_to_obj, t_table_tf)

            object_pos = teacher_env.unwrapped.sim.data.get_joint_qpos('object0:joint').copy()
            object_pos[:2] = t_obj_pose[:2]
            teacher_env.unwrapped.sim.data.set_joint_qpos('object0:joint', object_pos)
            teacher_env.unwrapped.sim.forward()

            teacher.reset()

            s_obs = student_env.unwrapped._get_obs()
            t_obs = teacher_env.unwrapped._get_obs()

        t_u = teacher.predict(t_obs)
        t_obs, _, done, _ = teacher_env.step(t_u)

        b_obs = dataset.b_scaler.transform(_flatten_obs(t_obs)[None], copy=True)[0]
        recon_t_obs = model.cross_decode_b_to_a(b_obs)
        recon_t_obs = dataset.a_scaler.inverse_transform(recon_t_obs[None], copy=True)[0]

        s_u = _controller(s_obs['observation'], recon_t_obs, prev_s_u, student_env.action_space)
        prev_s_u = s_u.copy()
        s_obs = student_env.step(s_u)[0]

        teacher_env.render()
        student_env.render()


if __name__ == '__main__':
    hand_to_yumi()
