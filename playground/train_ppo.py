import os
import glob
from datetime import datetime

import numpy as np
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines import PPO2

import gym
from gym.wrappers import FlattenDictWrapper
from gym.envs.robotics.fetch_env import goal_distance as fetch_env_goal_distance


ALG_NAME = 'ppo2'
OUT_DIR = f'../out/{ALG_NAME}'
REMOTE_OUT_DIR = f'/run/user/1000/gvfs/sftp:host=mordor.csc.kth.se,port=2222,user=carlora/home/carlora/thesis/repo/out/{ALG_NAME}'
os.makedirs(OUT_DIR, exist_ok=True)
current_epoch = None


def init_env(*, env_id, seed=0, env_kwargs=None, obs_keys=None, **_kwargs):
    def _init():
        kwargs = env_kwargs or dict()
        env = gym.make(env_id, **kwargs)
        env.seed(seed)
        if isinstance(env.unwrapped, gym.GoalEnv):
            dict_keys = obs_keys or ['observation', 'desired_goal']
            env = FlattenDictWrapper(env, dict_keys=dict_keys)
        env = Monitor(env, None)
        return env
    return _init


def unnormalize_obs(obs: np.ndarray, env: VecNormalize):
    return obs * np.sqrt(env.obs_rms.var + env.epsilon) + env.obs_rms.mean


def train(*, env_id, env_kwargs, ppo_params, steps, local_dir, seed=42, n_cpus=1, checkpoint_freq=1,
          obs_keys=None, custom_init=None, init_kwargs=None):

    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    local_dir = f'{local_dir}/{now}'
    checkpoints_dir = f'{local_dir}/checkpoints'
    normalizer_dir = f'{local_dir}/normalizer'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(normalizer_dir, exist_ok=True)

    is_fetch = 'Fetch' in env_id
    init_fn = custom_init if callable(custom_init) else init_env
    init_kwargs = init_kwargs or dict()
    env = SubprocVecEnv([init_fn(env_id=env_id, seed=seed+i, env_kwargs=env_kwargs,
                                 obs_keys=obs_keys, **init_kwargs) for i in range(n_cpus)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=200.)
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=local_dir, **ppo_params)

    global current_epoch
    current_epoch = -1

    def save_checkpoint(epoch):
        print('Saved checkpoint for epoch', epoch)
        model_path = f'{checkpoints_dir}/model_{epoch}.pkl'
        model.save(model_path)
        env.save_running_average(normalizer_dir)

    def callback(model_locals, model_globals):

        writer = model_locals.get('writer') # type: tf.summary.FileWriter
        update = model_locals.get('update') # type: int
        step = update * model.n_batch

        global current_epoch
        if current_epoch != update:
            current_epoch = update

            if current_epoch == 1 or (current_epoch % checkpoint_freq) == 0:
                save_checkpoint(current_epoch)

            obs = model_locals.get('obs')
            if is_fetch and obs is not None and len(obs) > 0:
                obs = unnormalize_obs(obs, env)
                achieved_goals, goals = obs[..., 3:6], obs[..., -3:]
                original_rewards = -fetch_env_goal_distance(achieved_goals, goals)
                avg_original_success_rate = np.mean(-original_rewards < 0.05)
                avg_original_rew = np.mean(original_rewards)

                s = tf.Summary()
                s.value.add(tag='custom/original_episode_reward', simple_value=avg_original_rew)
                s.value.add(tag='custom/original_success_rate', simple_value=avg_original_success_rate)
                writer.add_summary(s, step)

    model.learn(total_timesteps=steps, callback=callback, tb_log_name='tb')


def play(*, env_id, run_dir, env_kwargs=None, epoch=None, obs_keys=None, custom_init=None, init_kwargs=None):

    epoch = epoch or '*'
    model_path = '{run_dir}/checkpoints/model_{epoch}.pkl'
    normalizer_dir = f'{run_dir}/normalizer'

    if epoch == '*':
        paths = glob.glob(model_path.format(run_dir=run_dir, epoch='*'))
        if len(paths) == 0:
            raise FileNotFoundError
        epoch = np.max([int(p.split('model_')[1].split('.pkl')[0]) for p in paths])

    init_fn = custom_init if callable(custom_init) else init_env
    init_kwargs = init_kwargs or dict()
    env = DummyVecEnv([init_fn(env_id=env_id, seed=42, env_kwargs=env_kwargs, obs_keys=obs_keys, **init_kwargs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=200., training=False)
    env.load_running_average(normalizer_dir)

    model_path = model_path.format(run_dir=run_dir, epoch=epoch)
    model = PPO2.load(model_path, env=env)
    print(f'Loaded model for epoch {epoch}.')
    obs = env.reset()
    while True:
        env.render()
        u = model.predict(obs, deterministic=True)[0]
        obs, rew, done, info = env.step(u)
        print(rew)


def evaluate(*, env_id, run_dir, env_kwargs=None, epoch=None, obs_keys=None, n_steps=10_000, random_policy=False,
             custom_init=None, init_kwargs=None):

    epoch = epoch or '*'
    model_path = '{run_dir}/checkpoints/model_{epoch}.pkl'
    normalizer_dir = f'{run_dir}/normalizer'

    if epoch == '*':
        paths = glob.glob(model_path.format(run_dir=run_dir, epoch='*'))
        if len(paths) == 0:
            raise FileNotFoundError
        epoch = np.max([int(p.split('model_')[1].split('.pkl')[0]) for p in paths])

    init_fn = custom_init if callable(custom_init) else init_env
    init_kwargs = init_kwargs or dict()
    env = DummyVecEnv([init_fn(env_id=env_id, seed=42, env_kwargs=env_kwargs, obs_keys=obs_keys, **init_kwargs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=200., training=False)
    env.load_running_average(normalizer_dir)

    model_path = model_path.format(run_dir=run_dir, epoch=epoch)
    model = PPO2.load(model_path, env=env)
    print(f'Loaded model for epoch {epoch}.')
    obs = env.reset()
    total_rew = 0.0
    all_rews = []
    for i in range(n_steps):
        if random_policy:
            u = env.action_space.sample()
        else:
            u = model.predict(obs, deterministic=True)[0]
        obs, rew, done, info = env.step(u)
        all_rews.append(rew.item())
        total_rew += rew.item()
        rew_stdev = np.std(all_rews)
        rew_mean = total_rew / (i + 1)
        rew_stderr = rew_stdev / np.sqrt(i + 1)
        print(f'rew_mean: {rew_mean} (+- {rew_stderr})', )


def _ae_env_custom_init(*, env_kwargs, seed=None, **kwargs):

    import copy
    import _thesis_modules
    from playground.twin_vae import TwinVAE, SimpleAutoencoder
    from gym.agents.shadow_hand import HandPickAndPlaceAgent
    from gym.agents.yumi import YumiConstrainedAgent
    from gym.utils import transformations as gym_tf

    a_scaler = copy.deepcopy(env_kwargs['dataset'].a_scaler)
    b_scaler = copy.deepcopy(env_kwargs['dataset'].b_scaler)

    def _init():
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
            'YumiConstrained-v2'
        )

        teacher_agent = HandPickAndPlaceAgent(teacher_env)

        ae_model = TwinVAE.load(
            '../out/twin_yumi_hand_ae_resets_fixed_long/checkpoints/model_c1.pt',
            net_class=SimpleAutoencoder
        )
        # dataset = TwinDataset.load(f'{ms_thesis_path}/out/pp_yumi_hand_fixed_twin_dataset_3k.pkl')
        # dataset.normalize()

        s_table_tf = student_env.unwrapped.get_table_surface_pose()
        t_table_tf = teacher_env.unwrapped.get_table_surface_pose()

        def _student_obs_transformer(obs_dict):
            o = np.r_[obs_dict['observation'], obs_dict['desired_goal']]
            return a_scaler.transform(o[None], copy=True)[0]

        def _teacher_obs_transformer(obs_dict):
            o = np.r_[obs_dict['observation'], obs_dict['desired_goal']]
            return b_scaler.transform(o[None], copy=True)[0]

        def _sync_goals(*, t_env, s_env, **kwargs_):
            tf_to_goal = gym_tf.get_tf(np.r_[s_env.goal, 1., 0., 0., 0.], s_table_tf)
            t_goal_pose = gym_tf.apply_tf(tf_to_goal, t_table_tf)

            t_env.goal = np.r_[t_goal_pose[:3], np.zeros(4)]

            tf_to_obj = gym_tf.get_tf(s_env.get_object_pose(), s_table_tf)
            t_obj_pose = gym_tf.apply_tf(tf_to_obj, t_table_tf)

            object_pos = t_env.sim.data.get_joint_qpos('object:joint').copy()
            object_pos[:2] = t_obj_pose[:2]
            t_env.sim.data.set_joint_qpos('object:joint', object_pos)
            t_env.sim.forward()

            return dict(s_obs=s_env._get_obs(), t_obs=t_env._get_obs())

        gym_kwargs = dict(
            teacher_agent=teacher_agent,
            teacher_env=teacher_env,
            student_env=student_env,
            twin_ae_model=ae_model,
            student_is_a_env=True,
            student_obs_transform=_student_obs_transformer,
            teacher_obs_transform=_teacher_obs_transformer,
            sync_goals=_sync_goals,
        )

        env = gym.make('TwinAutoencoder-v0', **gym_kwargs)
        env.seed(seed)
        dict_keys = ['observation', 'desired_goal']
        env = FlattenDictWrapper(env, dict_keys=dict_keys)
        env = Monitor(env, None)
        return env
    return _init


def _ae_fetch_yumi_env_custom_init(*, env_kwargs, seed=None, target_task_is_push=False,
                                   imitation_rew_anneal_rate=0.0, **kwargs):

    import copy
    import _thesis_modules
    from playground.twin_vae import TwinVAE, SimpleAutoencoder
    from gym.agents.fetch import FetchPickAndPlaceAgent, FetchPushAgent
    from gym.agents.yumi import YumiConstrainedAgent
    from gym.utils import transformations as gym_tf

    a_scaler = copy.deepcopy(env_kwargs['dataset'].a_scaler)
    b_scaler = copy.deepcopy(env_kwargs['dataset'].b_scaler)

    init_task_rew_w = 1.0
    init_im_rew_w = 50.0

    def _init():
        teacher_env = gym.make(
            'FetchPickAndPlace-v1',
        )
        student_env = gym.make(
            'YumiConstrained-v2'
        )

        if target_task_is_push:
            teacher_agent = FetchPushAgent(teacher_env)
        else:
            teacher_agent = FetchPickAndPlaceAgent(teacher_env)

        ae_model = TwinVAE.load(
            '../out/twin_yumi_v2_fetch_ae_resets_new/checkpoints/model_c30.pt',
            net_class=SimpleAutoencoder
        )

        s_table_tf = student_env.unwrapped.get_table_surface_pose()
        t_table_tf = gym.make('HandPickAndPlace-v0').unwrapped.get_table_surface_pose()

        def _student_obs_transformer(obs_dict):
            o = np.r_[obs_dict['observation'], obs_dict['desired_goal']]
            return a_scaler.transform(o[None], copy=True)[0]

        def _teacher_obs_transformer(obs_dict):
            o = np.r_[obs_dict['observation'], obs_dict['desired_goal']]
            return b_scaler.transform(o[None], copy=True)[0]

        def _sync_goals(*, t_env, s_env, **kwargs_):

            if target_task_is_push:
                s_env.goal[2] = 0.025

            tf_to_goal = gym_tf.get_tf(np.r_[s_env.goal, 1., 0., 0., 0.], s_table_tf)
            t_goal_pose = gym_tf.apply_tf(tf_to_goal, t_table_tf)

            t_env.goal = t_goal_pose[:3]

            tf_to_obj = gym_tf.get_tf(s_env.get_object_pose(), s_table_tf)
            t_obj_pose = gym_tf.apply_tf(tf_to_obj, t_table_tf)

            object_pos = t_env.sim.data.get_joint_qpos('object0:joint').copy()
            object_pos[:2] = t_obj_pose[:2]
            t_env.sim.data.set_joint_qpos('object0:joint', object_pos)
            t_env.sim.forward()

            return dict(s_obs=s_env._get_obs(), t_obs=t_env._get_obs())

        def _rew_weight_update_rule(*, steps_since_init, **kwargs_):
            w = max(0.0, 1 - imitation_rew_anneal_rate * steps_since_init)
            im_w = init_im_rew_w * w
            task_w = init_task_rew_w * (1.0 - w)
            return max(im_w, 0.0), max(task_w, 0.0)

        if imitation_rew_anneal_rate == 0:
            _rew_weight_update_rule = None

        gym_kwargs = dict(
            teacher_agent=teacher_agent,
            teacher_env=teacher_env,
            student_env=student_env,
            twin_ae_model=ae_model,
            student_is_a_env=True,
            student_obs_transform=_student_obs_transformer,
            teacher_obs_transform=_teacher_obs_transformer,
            sync_goals=_sync_goals,
            task_rew_weight=init_task_rew_w,
            imitation_rew_weight=init_im_rew_w,
            rew_weight_update_rule=_rew_weight_update_rule,
        )

        env = gym.make('TwinAutoencoder-v0', **gym_kwargs)
        env.seed(seed)
        dict_keys = ['observation', 'desired_goal']
        env = FlattenDictWrapper(env, dict_keys=dict_keys)
        env = Monitor(env, None)
        return env
    return _init


if __name__ == '__main__':

    # train(
    #     env_id='YumiReachTwoArms-v0',
    #     env_kwargs=dict(reward_type='dense'),
    #     ppo_params=dict(
    #         n_steps=256
    #     ),
    #     steps=100_000_000,
    #     local_dir=f'{OUT_DIR}/yumi_reach_test4',
    #     n_cpus=15,
    #     checkpoint_freq=20,
    # )

    # play(
    #     env_id='FetchPickAndPlaceDense-v1',
    #     env_kwargs=dict(reward_params=dict(stepped=True), explicit_goal_distance=True),
    #     run_dir=glob.glob(f'{REMOTE_OUT_DIR}/fetch_stepped_rew_v3/*')[0],
    #     epoch=19520,
    # )

    # play(
    #     env_id='YumiReachLeftArm-v0',
    #     env_kwargs=dict(reward_type='dense'),
    #     run_dir=glob.glob(f'{REMOTE_OUT_DIR}/yumi_reach_test3/*')[0],
    #     epoch=None,
    # )

    # play(
    #     env_id='YumiReachTwoArms-v0',
    #     env_kwargs=dict(reward_type='dense'),
    #     run_dir=glob.glob(f'{REMOTE_OUT_DIR}/yumi_reach_test4/*')[0],
    #     epoch=None,
    # )

    # train(
    #     env_id='HandStepped-v0',
    #     env_kwargs=dict(),
    #     ppo_params=dict(),
    #     steps=5_000_000,
    #     local_dir=f'{OUT_DIR}/hand_stepped_test1',
    #     n_cpus=8,
    #     checkpoint_freq=10,
    #     obs_keys=['observation']
    # )

    # play(
    #     env_id='HandStepped-v0',
    #     env_kwargs=dict(render_substeps=True),
    #     run_dir=glob.glob(f'{REMOTE_OUT_DIR}/hand_stepped_test1/*')[0],
    #     epoch=None,
    #     obs_keys=['observation']
    # )

    # train(
    #     env_id='YumiStepped-v1',
    #     env_kwargs=dict(),
    #     ppo_params=dict(),
    #     steps=5_000_000,
    #     local_dir=f'{OUT_DIR}/yumi_stepped_test1',
    #     n_cpus=6,
    #     checkpoint_freq=5,
    # )

    # play(
    #     env_id='YumiStepped-v1',
    #     env_kwargs=dict(render_substeps=True),
    #     run_dir=glob.glob(f'{OUT_DIR}/yumi_stepped_test1/*')[0],
    #     epoch=None,
    # )

    # train(
    #     env_id='HandPickAndPlaceStepped-v0',
    #     env_kwargs=dict(),
    #     ppo_params=dict(),
    #     steps=5_000_000,
    #     local_dir=f'{OUT_DIR}/hand_stepped_pp_test1',
    #     n_cpus=7,
    #     checkpoint_freq=5,
    #     obs_keys=['observation', 'desired_goal']
    # )

    # play(
    #     env_id='HandPickAndPlaceStepped-v0',
    #     env_kwargs=dict(render_substeps=True),
    #     run_dir=glob.glob(f'{OUT_DIR}/hand_stepped_pp_test1/*')[0],
    #     epoch=175,
    #     obs_keys=['observation', 'desired_goal']
    # )

    # evaluate(
    #     env_id='HandPickAndPlaceStepped-v0',
    #     env_kwargs=dict(render_substeps=False),
    #     run_dir=glob.glob(f'{OUT_DIR}/hand_stepped_pp_test1/*')[0],
    #     epoch=175,
    #     obs_keys=['observation', 'desired_goal'],
    #     random_policy=True
    # )

    import _thesis_modules
    from playground.twin_vae import TwinDataset
    # dataset_ = TwinDataset.load(f'../out/pp_yumi_hand_fixed_twin_dataset_3k.pkl')
    dataset_ = TwinDataset.load(f'../out/pp_yumi_v2_fetch_twin_dataset_10k.pkl')
    dataset_.normalize()

    # train(
    #     env_id='TwinAutoencoder-v0',
    #     env_kwargs=dict(dataset=dataset_),
    #     ppo_params=dict(),
    #     steps=5_000_000,
    #     local_dir=f'{OUT_DIR}/ae_fetch_yumi_push_env_test1',
    #     n_cpus=4,
    #     checkpoint_freq=5,
    #     custom_init=_ae_fetch_yumi_env_custom_init,
    #     init_kwargs=dict(target_task_is_push=True, imitation_rew_anneal_rate=1/200_000)
    # )

    play(
        env_id='TwinAutoencoder-v0',
        env_kwargs=dict(dataset=dataset_),
        run_dir=glob.glob(f'{OUT_DIR}/ae_fetch_yumi_push_env_test1/*')[1],
        custom_init=_ae_fetch_yumi_env_custom_init,
        init_kwargs=dict(target_task_is_push=True),
    )

    # train(
    #     env_id='TwinAutoencoder-v0',
    #     env_kwargs=dict(dataset=dataset_),
    #     ppo_params=dict(),
    #     steps=5_000_000,
    #     local_dir=f'{OUT_DIR}/ae_fetch_yumi_env_test1',
    #     n_cpus=4,
    #     checkpoint_freq=5,
    #     custom_init=_ae_fetch_yumi_env_custom_init
    # )

    # play(
    #     env_id='TwinAutoencoder-v0',
    #     env_kwargs=dict(dataset=dataset_),
    #     run_dir=glob.glob(f'{OUT_DIR}/ae_fetch_yumi_env_test1/*')[1],
    #     custom_init=_ae_fetch_yumi_env_custom_init,
    # )

    # train(
    #     env_id='TwinAutoencoder-v0',
    #     env_kwargs=dict(dataset=dataset_),
    #     ppo_params=dict(),
    #     steps=5_000_000,
    #     local_dir=f'{OUT_DIR}/ae_env_test1',
    #     n_cpus=4,
    #     checkpoint_freq=5,
    #     custom_init=_ae_env_custom_init
    # )

    # play(
    #     env_id='TwinAutoencoder-v0',
    #     env_kwargs=dict(dataset=dataset_),
    #     run_dir=glob.glob(f'{OUT_DIR}/ae_env_test1/*')[0],
    #     custom_init=_ae_env_custom_init,
    # )
