from time import sleep

import gym
import pybullet as pb
import pybullet_data
import pybulletgym.envs


if __name__ == '__main__':

    env = gym.make("HumanoidPyBulletEnv-v0")
    env.render(mode="human")
    env.reset()

    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -10)
    planeId = pb.loadURDF("plane.urdf", basePosition=[0, 0, 0.01])

    while True:
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        sleep(1/60)
