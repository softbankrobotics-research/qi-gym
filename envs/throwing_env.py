#!/usr/bin/env python
# coding: utf-8

import sys

CV2_ROS = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if CV2_ROS in sys.path:
    sys.path.remove(CV2_ROS)
    sys.path.append(CV2_ROS)

import os
import gym
import time
import argparse
import stable_baselines
from datetime import datetime
import numpy as np
from gym import spaces
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec
from stable_baselines import DDPG

import pybullet
import pybullet_data
from qibullet import PepperVirtual
from qibullet import SimulationManager
from urllib.request import Request, urlopen


class RobotEnv(gym.Env):
    def __init__(self, gui=False):
        self.r_kinematic_chain = [
            # "KneePitch",
            # "HipPitch",
            # "HipRoll",
            "RShoulderPitch",
            "RShoulderRoll",
            "RElbowRoll",
            "RElbowYaw",
            "RWristYaw"]

        self.initial_stand = [
            1.207,
            -0.129,
            1.194,
            1.581,
            1.632]

        self.gui = gui

        self.joints_initial_pose = list()

        # Passed to True at the end of an episode
        self.episode_start_time = None
        self.episode_over = False
        self.episode_failed = False
        self.episode_reward = 0.0
        self.episode_number = 0
        self.episode_steps = 0

        self.simulation_manager = SimulationManager()

        # self.initial_bucket_pose = [0.65, -0.2, 0.0]
        self.projectile_radius = 0.03

        self._setupScene()

        lower_limits = list()
        upper_limits = list()

        # Bucket footprint in base_footprint and r_gripper 6D in base_footprint
        # (in reality in odom, but the robot won't move and is on the odom
        # frame): (x_b, y_b, x, y, z, rx, ry, rz)
        # lower_limits.extend([-10, -10, -10, -10, 0, -7, -7, -7])
        # upper_limits.extend([10, 10, 10, 10, 10, 7, 7, 7])
        lower_limits.extend([-10, -10])
        upper_limits.extend([10, 10])

        # Add the joint positions to the limits
        lower_limits.extend([self.pepper.joint_dict[joint].getLowerLimit() for
                            joint in self.r_kinematic_chain])
        upper_limits.extend([self.pepper.joint_dict[joint].getUpperLimit() for
                            joint in self.r_kinematic_chain])

        # Add gripper position to the limits
        lower_limits.extend([-2, -2, 0, -1, -1, -1, -1])
        upper_limits.extend([2, 2, 3, 1, 1, 1, 1])

        self.observation_space = spaces.Box(
            low=np.array(lower_limits),
            high=np.array(upper_limits))

        # Define the action space
        velocity_limits = [
            self.pepper.joint_dict[joint].getMaxVelocity() for
            joint in self.r_kinematic_chain]
        velocity_limits.extend([
            -self.pepper.joint_dict[joint].getMaxVelocity() for
            joint in self.r_kinematic_chain])

        normalized_limits = self.normalize(velocity_limits)
        self.max_velocities = normalized_limits[:len(self.r_kinematic_chain)]
        self.min_velocities = normalized_limits[len(self.r_kinematic_chain):]

        self.action_space = spaces.Box(
            low=np.array(self.min_velocities),
            high=np.array(self.max_velocities))

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        try:
            action = list(action)
            assert len(action) == len(self.action_space.high.tolist())

        except AssertionError:
            print("Incorrect action")
            return None, None, None, None

        self.episode_steps += 1
        np.clip(action, self.min_velocities, self.max_velocities)
        self._setVelocities(self.r_kinematic_chain, action)

        obs, reward = self._getState()
        return obs, reward, self.episode_over, {}

    def reset(self):
        """
        Resets the environment for a new episode
        """
        self.episode_over = False
        self.episode_failed = False
        self.episode_reward = 0.0
        self.episode_steps = 0
        self._resetScene()

        # Reset the start time for the current episode
        self.episode_start_time = time.time()

        # Fill and return the observation
        return self._getObservation()

    def render(self, mode='human', close=False):
        pass

    def _setVelocities(self, angles, normalized_velocities):
        """
        Sets velocities on the robot joints
        """
        for angle, velocity in zip(angles, normalized_velocities):
            # Unnormalize the velocity
            velocity *= self.pepper.joint_dict[angle].getMaxVelocity()

            position = self.pepper.getAnglesPosition(angle)
            lower_limit = self.pepper.joint_dict[angle].getLowerLimit()
            upper_limit = self.pepper.joint_dict[angle].getUpperLimit()

            if position <= lower_limit and velocity < 0.0:
                velocity = 0.0
                self.episode_failed = True
            elif position >= upper_limit and velocity > 0.0:
                velocity = 0.0
                self.episode_failed = True

            pybullet.setJointMotorControl2(
                self.pepper.robot_model,
                self.pepper.joint_dict[angle].getIndex(),
                pybullet.VELOCITY_CONTROL,
                targetVelocity=velocity,
                force=self.pepper.joint_dict[angle].getMaxEffort())

    def _getBucketPosition(self):
        """
        Returns the position of the target bucket in the world
        """
        # Get the position of the bucket (goal) in the world
        bucket_pose, bucket_qrot = pybullet.getBasePositionAndOrientation(
            self.bucket)

        return bucket_pose, bucket_qrot

    def _getProjectilePosition(self):
        """
        Returns the position of the projectile in the world
        """
        # Get the position of the projectile in the world
        project_pose, project_qrot = pybullet.getBasePositionAndOrientation(
            self.projectile)

        return project_pose, project_qrot

    def _getLinkPosition(self, link_name):
        """
        Returns the position of the link in the world frame
        """
        link_state = pybullet.getLinkState(
            self.pepper.robot_model,
            self.pepper.link_dict[link_name].getIndex())

        return link_state[0], link_state[1]

    def _computeProjectileSpawnPose(self):
        """
        Returns the ideal position for the projectile (in the robot's hand)
        """
        r_wrist_pose, _ = self._getLinkPosition("r_wrist")
        return [
            r_wrist_pose[0] + 0.04,
            r_wrist_pose[1] - 0.01,
            r_wrist_pose[2] + 0.064]

    def _computeBucketSpawnPose(self):
        """
        Returns a spawning pose for the targeted bin
        """
        # Ranges in polar coordinates
        radius_range = [0.4, 0.65]
        angle_range = [-1.2, 0.4]
        radius = np.random.uniform(low=radius_range[0], high=radius_range[1])
        angle = np.random.uniform(low=angle_range[0], high=angle_range[1])

        # return [radius * np.cos(angle), radius * np.sin(angle), 0.0]
        return [0.6, -0.22, 0.0]

    def normalize(self, values, range_min=-1.0, range_max=1.0):
        """
        Normalizes values (list) according to a specific range
        """
        zero_bound = [x - min(values) for x in values]
        range_bound = [
            x * (range_max - range_min) / (max(zero_bound) - min(zero_bound))
            for x in zero_bound]

        return [x - max(range_bound) + range_max for x in range_bound]

    def _getState(self, convergence_norm=0.15):
        """
        Gets the observation and computes the current reward. Will also
        determine if the episode is over or not, by filling the episode_over
        boolean. When the euclidian distance between the wrist link and the
        cube is inferior to the one defined by the convergence criteria, the
        episode is stopped
        """
        reward = 0.0

        # Get position of the object and gripper pose in the odom frame
        projectile_pose, _ = self._getProjectilePosition()
        bucket_pose, _ = self._getBucketPosition()

        # Check if there is a self collision on the r_wrist, if so stop the
        # episode. Else, check if the ball is below 0.049 (height of the
        # trash's floor)
        if self.pepper.isSelfColliding("r_wrist") or\
                self.pepper.isSelfColliding("RForeArm"):
            self.episode_over = True
            self.episode_failed = True
            reward += -1
        elif projectile_pose[2] <= 0.049:
            self.episode_over = True
        elif (time.time() - self.episode_start_time) > 2.5:
            self.episode_over = True
            self.episode_failed = True
            reward += -1

        # Fill the observation
        obs = self._getObservation()

        # Compute the reward
        bucket_footprint = [bucket_pose[0], bucket_pose[1], 0.0]
        projectile_footprint = [projectile_pose[0], projectile_pose[1], 0.0]

        previous_footprint = [
            self.prev_projectile_pose[0],
            self.prev_projectile_pose[1],
            0.0]

        prev_to_target =\
            np.array(bucket_footprint) - np.array(previous_footprint)
        current_to_target =\
            np.array(bucket_footprint) - np.array(projectile_footprint)

        # test
        # init_to_current = np.array(projectile_footprint) - np.array([
        #     self.initial_projectile_pose[0],
        #     self.initial_projectile_pose[1],
        #     0.0])

        norm_to_target = np.linalg.norm(current_to_target)
        reward += np.linalg.norm(prev_to_target) - norm_to_target

        # If the episode is over, check the position of the floor projection
        # of the projectile, to know wether if it's in the trash or not
        if self.episode_over:
            if norm_to_target <= 0.115 and not self.episode_failed:
                reward += 2.0
            else:
                self.episode_failed = True

            initial_proj_footprint = [
                self.initial_projectile_pose[0],
                self.initial_projectile_pose[1],
                0.0]

            initial_norm = np.linalg.norm(
                np.array(initial_proj_footprint) - np.array(bucket_footprint))

            # Test replace norm
            reward += initial_norm - norm_to_target

        # Test velocity vector reward
        # K = 500

        # if (norm_to_target <= 0.115 and projectile_pose[2] <= 0.32):
        #     reward += 1/K
        # elif np.linalg.norm(init_to_current) != 0:
        #     ref_cos = np.cos(np.arctan2(0.3, norm_to_target))
        #     center_cos =\
        #         np.dot(current_to_target, init_to_current) /\
        #         (norm_to_target * np.linalg.norm(init_to_current))

        #     if center_cos > ref_cos:
        #         reward += 1/K
        #     else:
        #         reward += center_cos/K
        # else:
        #     reward += -1/K

        # Add the reward to the episode reward
        self.episode_reward += reward

        # Update the previous projectile pose
        self.prev_projectile_pose = projectile_pose

        if self.episode_over:
            self.episode_number += 1
            self._printEpisodeSummary()

        return obs, reward

    def _getObservation(self):
        """
        Returns the observation

        Returns:
            obs - the list containing the observations
        """
        # Get position of the projectile and bucket in the odom frame
        # projectile_pose, _ = self._getProjectilePosition()
        bucket_pose, _ = self._getBucketPosition()

        # Get the position of the r_gripper in the odom frame (base footprint
        # is on the origin of the odom frame in the xp)
        gripper_pose, gripper_rot = self._getLinkPosition("r_gripper")

        # Fill and return the observation
        return [pose for pose in bucket_pose[:2]] +\
            self.pepper.getAnglesPosition(self.r_kinematic_chain) +\
            [pose for pose in gripper_pose] +\
            [rot for rot in gripper_rot]

    def _setupScene(self):
        """
        Setup a scene environment within the simulation
        """
        self.client = self.simulation_manager.launchSimulation(gui=self.gui)
        self.pepper = self.simulation_manager.spawnPepper(
            self.client,
            spawn_ground_plane=True)

        self.pepper.goToPosture("Stand", 1.0)
        self.pepper.setAngles("RHand", 0.7, 1.0)
        self.pepper.setAngles(
            self.r_kinematic_chain,
            self.initial_stand,
            1.0)

        time.sleep(1.0)
        self.joints_initial_pose = self.pepper.getAnglesPosition(
            self.pepper.joint_dict.keys())

        pybullet.setAdditionalSearchPath("environment")
        self.bucket = pybullet.loadURDF(
            "trash.urdf",
            self._computeBucketSpawnPose(),
            flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL)

        # The initial pose of the projectile
        self.initial_projectile_pose = self._computeProjectileSpawnPose()

        self.projectile = pybullet.loadURDF(
            "ball.urdf",
            self.initial_projectile_pose)

        time.sleep(0.2)

        # Get position of the projectile in the odom frame
        self.prev_projectile_pose, _ = self._getProjectilePosition()

    def _resetScene(self):
        """
        Resets the scene for a new scenario
        """
        pybullet.resetBasePositionAndOrientation(
            self.pepper.robot_model,
            posObj=[0.0, 0.0, 0.0],
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client)

        self._hardResetJointState()

        # The initial pose of the projectile
        self.initial_projectile_pose = self._computeProjectileSpawnPose()

        pybullet.resetBasePositionAndOrientation(
            self.projectile,
            posObj=self.initial_projectile_pose,
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client)
        pybullet.resetBasePositionAndOrientation(
            self.bucket,
            posObj=self._computeBucketSpawnPose(),
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client)

        time.sleep(0.2)

        # Get position of the object and gripper pose in the odom frame
        self.prev_projectile_pose, _ = self._getProjectilePosition()

    def _hardResetJointState(self):
        """
        Performs a hard reset on the joints of the robot, avoiding the robot to
        get stuck in a position
        """
        for joint, position in\
                zip(self.pepper.joint_dict.keys(), self.joints_initial_pose):
            pybullet.setJointMotorControl2(
                self.pepper.robot_model,
                self.pepper.joint_dict[joint].getIndex(),
                pybullet.VELOCITY_CONTROL,
                targetVelocity=0.0)
            pybullet.resetJointState(
                    self.pepper.robot_model,
                    self.pepper.joint_dict[joint].getIndex(),
                    position)

    def _printEpisodeSummary(self, info_dict={}):
        """
        Prints a summary for an episode

        Parameters:
            info_dict - Dictionnary containing extra data to be displayed
        """
        if self.episode_failed:
            episode_status = "FAILURE"
        else:
            episode_status = "SUCCESS"

        print("#---------Episode-Summary---------#")
        print("Episode number: " + str(self.episode_number))
        print("Episode's number of steps: " + str(self.episode_steps))
        print("Episode status: " + episode_status)
        print("Episode reward: " + str(self.episode_reward))

        for key, value in info_dict.items():
            print(key + ": " + str(value))

    def _termination(self):
        """
        Terminates the environment
        """
        self.simulation_manager.stopSimulation(self.client)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--generate_pretrain",
        type=int,
        default=0,
        help="If true, launch an interface to generate an expert trajectory")

    parser.add_argument(
        "--train",
        type=int,
        default=1,
        help="True: training, False: using a trained model")

    parser.add_argument(
        "--algo",
        type=str,
        default="ppo2",
        help="The learning algorithm to be used (ppo2 or ddpg)")

    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="The version name of the model")

    parser.add_argument(
        "--gui",
        type=int,
        default=1,
        help="Wether the GUI of the simulation should be used or not. 0 or 1")

    args = parser.parse_args()
    algo = args.algo.lower()

    try:
        assert args.gui == 0 or args.gui == 1
        assert algo == "ppo2" or algo == "ddpg"

    except AssertionError as e:
        print(str(e))
        return

    env = RobotEnv(gui=args.gui)
    vec_env = DummyVecEnv([lambda: env])

    # Generate an expert trajectory
    if args.generate_pretrain:
        pass
    
    # Train a model
    elif args.train == 1:
        while True:
            req = Request(
                "https://frightanic.com/goodies_content/docker-names.php",
                headers={'User-Agent': 'Mozilla/5.0'})

            webpage = str(urlopen(req).read())
            word = webpage.split("b\'")[1]
            word = word.split("\\")[0]
            word.replace(" ", "_")

            try:
                assert os.path.isfile(
                    "models/" + algo + "_throw_" + word + ".pkl")

            except AssertionError:
                break

        log_name = "./logs/throw/" + word

        if algo == "ppo2":
            # For recurrent policies, nminibatches should be a multiple of the 
            # nb of env used in parallel (so for LSTM, 1)
            model = PPO2(
                MlpLstmPolicy,
                vec_env,
                nminibatches=1,
                verbose=0,
                tensorboard_log=log_name)

        elif algo == "ddpg":
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(env.action_space.shape[-1]),
                sigma=float(0.5) * np.ones(env.action_space.shape[-1]))

            model = DDPG(
                stable_baselines.ddpg.LnMlpPolicy,
                env,
                verbose=0,
                param_noise=None,
                action_noise=action_noise,
                tensorboard_log=log_name)

        try:
            model.learn(total_timesteps=1000000)
        
        except KeyboardInterrupt:
            print("#---------------------------------#")
            print("Training \'" + word + "\' interrupted")
            print("#---------------------------------#")
            sys.exit(1)

        model.save("models/" + algo + "_throw_" + word)

    # Use a trained model
    else:
        if args.model == "":
            print("Specify the version of the model using --model")
            return

        if algo == "ppo2":
            model = PPO2.load("models/" + algo + "_throw_" + args.model)
        elif algo == "ddpg":
            model = DDPG.load("models/" + algo + "_throw_" + args.model)

        for test in range(10):
            dones = False
            obs = env.reset()

            while not dones:
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)

    time.sleep(2)
    env._termination()


if __name__ == "__main__":
    main()
