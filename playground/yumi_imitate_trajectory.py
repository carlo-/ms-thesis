import numpy as np
import gym
from gym.agents.yumi import YumiConstrainedAgent
from gym.agents.shadow_hand import HandPickAndPlaceAgent
from gym.agents.fetch import FetchPickAndPlaceAgent, FetchPushAgent
from gym.utils import transformations as tf

import _thesis_modules
from playground.twin_vae import TwinVAE, TwinDataset, SimpleAutoencoder, VAE


def _controller(s_obs_, t_obs_, prev_s_u_, s_action_space_, yumi_env_version=2) -> np.ndarray:
    u = np.zeros(s_action_space_.shape)
    pos_err = t_obs_[:3] - s_obs_[:3]
    t_g_dist = np.linalg.norm(t_obs_[6:9] - t_obs_[9:12], ord=2)
    # s_g_dist = np.linalg.norm(s_obs_['observation'][6:9] - s_obs_['observation'][9:12], ord=2)
    # g_dist_err = t_g_dist - s_g_dist
    if yumi_env_version == 2:
        pos_k = 10.0
        grasp_k = 1.0
        grasp_range = [0.05, 0.20]
    else:
        pos_k = 5.0
        grasp_k = 1.2
        grasp_range = [0.05, 0.30]
    u[1:4] = pos_err * pos_k
    u[0] = np.interp(t_g_dist, grasp_range, [-1, 1]) * grasp_k
    return np.clip(u, -1, 1)


def _flatten_obs(obs_dict):
    return np.r_[obs_dict['observation'], obs_dict['desired_goal']]


def yumi_to_yumi(teacher_eps=None, env_v=2):

    if isinstance(teacher_eps, bool):
        if teacher_eps:
            teacher_eps = TwinDataset.load('../out/pp_yumi_v2_fetch_twin_dataset_10k.pkl').a_episodes
        else:
            teacher_eps = None

    teacher_env = gym.make(
        f'YumiConstrained-v{env_v}',
        reward_type='sparse',
        render_poses=False,
    )

    student_env = gym.make(
        f'YumiConstrained-v{env_v}',
        reward_type='sparse',
        render_poses=False,
    )

    teacher = YumiConstrainedAgent(teacher_env)

    done = True
    t_obs = s_obs = prev_s_u = None
    ep_i = -1
    step_i = 0

    while True:

        if done:
            ep_i += 1
            step_i = 0

            student_env.reset()
            prev_s_u = np.zeros(student_env.action_space.shape)

            if teacher_eps is None:
                teacher_env.reset()
                goal = teacher_env.unwrapped.sim_env.goal
                obj_init_xy_pos = teacher_env.unwrapped.sim.data.get_joint_qpos('object0:joint')[:2]
                t_obs = teacher_env.unwrapped._get_obs()
            else:
                goal = teacher_eps[ep_i][0][-3:]
                obj_init_xy_pos = teacher_eps[ep_i][0][18:20]

            student_env.unwrapped.sim_env.goal = goal.copy()
            object_pos = student_env.unwrapped.sim.data.get_joint_qpos('object0:joint').copy()
            object_pos[:2] = obj_init_xy_pos.copy()
            student_env.unwrapped.sim.data.set_joint_qpos('object0:joint', object_pos)
            student_env.unwrapped.sim.forward()
            s_obs = student_env.unwrapped._get_obs()

        if teacher_eps is None:
            t_u = teacher.predict(t_obs)
            t_obs, _, done, _ = teacher_env.step(t_u)
            t_obs_np = t_obs['observation']
        else:
            t_obs_np = teacher_eps[ep_i][step_i][:-3]
            done = step_i + 1 == len(teacher_eps[ep_i])

        s_u = _controller(s_obs['observation'], t_obs_np, prev_s_u, student_env.action_space, env_v)
        prev_s_u = s_u.copy()
        s_obs = student_env.step(s_u)[0]

        if teacher_eps is None:
            teacher_env.render()
        student_env.render()

        step_i += 1


def yumi_to_yumi_recon():

    env_v = 2

    teacher_env = gym.make(
        f'YumiConstrained-v{env_v}',
        reward_type='sparse',
        render_poses=False,
    )

    student_env = gym.make(
        f'YumiConstrained-v{env_v}',
        reward_type='sparse',
        render_poses=False,
    )

    teacher = YumiConstrainedAgent(teacher_env)

    model = TwinVAE.load('../out/twin_yumi_hand_ae_resets_fixed_long/checkpoints/model_c1.pt',
                         net_class=SimpleAutoencoder)
    dataset = TwinDataset.load('../out/pp_yumi_hand_fixed_twin_dataset_3k.pkl')

    # model = TwinVAE.load('../out/twin_yumi_hand_ae_resets_goal_init/checkpoints/model_c14.pt',
    #                      net_class=SimpleAutoencoder)
    # dataset = TwinDataset.load('../out/pp_yumi_twin_dataset_3k.pkl')

    dataset.normalize()

    import torch

    def decode_a_to_a(a_obs_):
        with torch.no_grad():
            a = torch.tensor(a_obs_[None], dtype=torch.float32, device=model.device)
            z = model.a_vae.encode(a, reparameterize=False)
            recon_a = model.a_vae.decode(z)[0]
        return recon_a.cpu().numpy()

    done = True
    t_obs = s_obs = prev_s_u = None

    # o_sum = np.zeros(3)
    # o_sum_c = 0

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

        a_obs = dataset.a_scaler.transform(_flatten_obs(t_obs)[None], copy=True)[0]
        recon_t_obs = decode_a_to_a(a_obs)
        recon_t_obs = dataset.a_scaler.inverse_transform(recon_t_obs[None], copy=True)[0]

        # o_sum += recon_t_obs[:3] - t_obs['observation'][:3]
        # o_sum_c += 1
        # print(o_sum / o_sum_c)

        s_u = _controller(s_obs['observation'], recon_t_obs, prev_s_u, student_env.action_space, env_v)
        prev_s_u = s_u.copy()
        s_obs = student_env.step(s_u)[0]

        teacher_env.render()
        student_env.render()


def hand_to_yumi():

    # model = TwinVAE.load('../out/pp_and_reach_yumi_twin_ae_test_z15/checkpoints/model_c49.pt',
    #                      net_class=SimpleAutoencoder)

    # model = TwinVAE.load('../out/twin_ae_kdl_test/checkpoints/model_c12.pt',
    #                      net_class=SimpleAutoencoder)

    # model = TwinVAE.load('../out/twin_yumi_hand_ae_resets/checkpoints/model_c6.pt',
    #                      net_class=SimpleAutoencoder)

    # model = TwinVAE.load('../out/twin_yumi_hand_ae_resets_goal_init/checkpoints/model_c14.pt',
    #                      net_class=SimpleAutoencoder)

    model = TwinVAE.load('../out/twin_yumi_hand_ae_resets_fixed_goal_init/checkpoints/model_c0.pt',
                         net_class=SimpleAutoencoder)

    # model = TwinVAE.load('../out/twin_vae_resets_test/checkpoints/model_c5.pt',
    #                      net_class=VAE)

    # dataset = TwinDataset.load('../out/pp_yumi_twin_dataset_3k.pkl')
    dataset = TwinDataset.load('../out/pp_yumi_hand_fixed_twin_dataset_3k.pkl')
    # dataset = TwinDataset.merge(
    #     TwinDataset.load('../out/pp_yumi_twin_dataset_3k.pkl'),
    #     TwinDataset.load('../out/pp_reach_yumi_twin_dataset_2k.pkl')
    # )
    dataset.normalize()

    teacher_env = gym.make(
        'HandPickAndPlace-v0',
        ignore_rotation_ctrl=True,
        ignore_target_rotation=True,
        success_on_grasp_only=False,
        randomize_initial_arm_pos=False,
        randomize_initial_object_pos=True,
        object_id='box'
    )

    student_env = gym.make(
        'YumiConstrained-v2',
        reward_type='sparse',
        render_poses=False,
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

        s_u = _controller(s_obs['observation'], recon_t_obs, prev_s_u, student_env.action_space, yumi_env_version=2)
        prev_s_u = s_u.copy()
        s_obs = student_env.step(s_u)[0]

        teacher_env.render()
        student_env.render()


def fetch_to_yumi_agent():
    model = TwinVAE.load('../out/twin_yumi_v2_fetch_ae_resets_new_l/checkpoints/model_c1.pt',
                         net_class=SimpleAutoencoder)
    dataset = TwinDataset.load('../out/pp_yumi_v2_fetch_twin_dataset_10k.pkl')
    dataset.normalize()

    teacher_env = gym.make(
        'FetchPickAndPlace-v1',
        reward_type='sparse'
    )

    student_env = gym.make(
        'YumiConstrained-v2',
        reward_type='sparse',
        render_poses=False,
        object_on_table=True,
    )

    from gym.agents.yumi import YumiImitatorAgent

    # teacher = FetchPickAndPlaceAgent(teacher_env)
    teacher = FetchPushAgent(teacher_env)
    agent = YumiImitatorAgent(student_env, teacher_env=teacher_env, teacher_agent=teacher, a_scaler=dataset.a_scaler,
                              b_scaler=dataset.b_scaler, model=model)

    done = True
    obs = None

    while True:
        if done:
            obs = student_env.reset()
            agent.reset()

        u = agent.predict(obs)
        obs, rew, done, info = student_env.step(u)

        if done:
            if info['is_success'] == 1:
                print('yey')
            else:
                print('ney')

        # student_env.render()
        # teacher_env.render()


def fetch_to_yumi():

    # model = TwinVAE.load('../out/twin_yumi_fetch_ae_resets/checkpoints/model_c3.pt',
    #                      net_class=SimpleAutoencoder)
    # dataset = TwinDataset.load('../out/pp_yumi_fetch_twin_dataset_5k.pkl')

    model = TwinVAE.load('../out/twin_yumi_v2_fetch_ae_resets_new_l/checkpoints/model_c1.pt',
                         net_class=SimpleAutoencoder)
    dataset = TwinDataset.load('../out/pp_yumi_v2_fetch_twin_dataset_10k.pkl')

    dataset.normalize()

    teacher_env = gym.make(
        'FetchPickAndPlace-v1',
        reward_type='sparse'
    )

    student_env = gym.make(
        'YumiConstrained-v2',
        reward_type='sparse',
        render_poses=False,
    )

    # teacher = FetchPickAndPlaceAgent(teacher_env)
    teacher = FetchPushAgent(teacher_env)

    done = True
    t_obs = s_obs = prev_s_u = None

    s_table_tf = student_env.unwrapped.get_table_surface_pose()
    t_table_tf = gym.make('HandPickAndPlace-v0').unwrapped.get_table_surface_pose()

    while True:

        if done:
            student_env.reset()
            teacher_env.reset()
            prev_s_u = np.zeros(student_env.action_space.shape)

            student_env.unwrapped.goal[2] = 0.025

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

        s_u = _controller(s_obs['observation'], recon_t_obs, prev_s_u, student_env.action_space, yumi_env_version=2)
        prev_s_u = s_u.copy()
        s_obs = student_env.step(s_u)[0]

        teacher_env.render()
        student_env.render()


if __name__ == '__main__':

    def foo_():
        import pickle
        file_path = "/home/carlo/KTH/thesis/misc/hindsight-experience-replay/demonstrations/yumi_imitator_from_fetch_push_100.pkl"
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        n_eps, ep_len = data['mb_obs'].shape[:2]
        all_eps = []
        for e in range(n_eps):
            ep = []
            for s in range(ep_len-1):
                obs = data['mb_obs'][e][s]
                ag = data['mb_ag'][e][s]
                g = data['mb_g'][e][s]
                ep.append(_flatten_obs(dict(
                    observation=obs,
                    achieved_goal=ag,
                    desired_goal=g,
                )))
            all_eps.append(ep)
        return all_eps

    t_eps_ = foo_()
    yumi_to_yumi(teacher_eps=t_eps_)
    # fetch_to_yumi_agent()
    # yumi_to_yumi(teacher_eps=False)
    # fetch_to_yumi()
    # hand_to_yumi()
    # yumi_to_yumi_recon()
