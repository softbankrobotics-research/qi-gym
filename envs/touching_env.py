#!/usr/bin/env python
# coding: utf-8

import sys
import os
import time
import gym
import argparse
import numpy as np
import pybullet
import pybullet_data

from gym import spaces
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from qibullet import PepperVirtual
from qibullet import SimulationManager
from urllib.request import Request, urlopen

CV2_ROS = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if CV2_ROS in sys.path:
    sys.path.remove(CV2_ROS)
    sys.path.append(CV2_ROS)

import stable_baselines
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines import PPO2
from stable_baselines import DDPG

EPOCHS = 4
EPISODES = 10000000
DETERMINISTIC = False


class CustomDDPGPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDDPGPolicy, self).__init__(
            *args, **kwargs,
            layers=[512, 512],
            layer_norm=False,
            feature_extraction="mlp")


class RobotEnv(gym.Env):
    SIMULATION_STEP = 1.0 / 240.0
    EPISODE_TIME = 5.0
    HIP_PITCH_MAX = 0.05
    HIP_PITCH_MIN = -0.4
    TARGET_MAX = 1.5
    TARGET_MIN = -1.5
    HEAD_REWARD_WEIGHT = 0.25
    HAND_REWARD_WEIGHT = 0.75
    JOINTS_VELOCITY = 0.15
    HEAD_VELOCITY = 0.7

    def __init__(self, gui=False):
        """
        Constructor
        """

        # Joint's names to use on the movement
        self.r_kinematic_chain = [
            "HipPitch",
            "RShoulderPitch",
            "RShoulderRoll",
            "RElbowRoll",
            "HeadYaw",
            "HeadPitch"
            ]

        self.gui = gui
        self.joints_initial_pose = list()

        # Passed to True at the end of an episode
        self.episode_over = False
        self.episode_failed = False

        self.episode_start_time = None
        self.episode_reward = 0.0
        self.episode_number = 0
        self.episode_steps = 0
        self.limit_max = 0

        self.hands_norm = 0
        self.orientations_norm = 0

        self.simulation_manager = SimulationManager()
        self._setupScene()

        pybullet.setRealTimeSimulation(0)
        pybullet.setTimeStep(self.SIMULATION_STEP)

        # Define the observations space
        lower_limits = list()
        upper_limits = list()

        # Add the limits of the observations, normalized betwen [-1, 1]:
        #   the number of joint angles: len(self.r_kinematic_chain)
        #   3D coordinate of the robot hand
        #   3D coordinate of the target
        #   3D coordinate of the direction vector Head-Target
        #   3D coordinate of the direction vector of the Head orientation
        size_coordinates = 12
        size_observations = len(self.r_kinematic_chain) + size_coordinates
        lower_limits.extend(np.full((size_observations), -1, dtype=float))
        upper_limits.extend(np.full((size_observations), 1, dtype=float))

        self.observation_space = spaces.Box(
            low=np.array(lower_limits),
            high=np.array(upper_limits))

        # Define the action space for the kinematic chain
        velocity_limits = [
            self.pepper.joint_dict[joint].getMaxVelocity() for
            joint in self.r_kinematic_chain]
        self.limit_max = max(velocity_limits)
        velocity_limits.extend([
            -self.pepper.joint_dict[joint].getMaxVelocity() for
            joint in self.r_kinematic_chain])

        # Normalize the action space
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
            action : the action to perform returned by the model

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :
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
            pybullet.stepSimulation()
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

    def _setVelocities(self, joints, normalized_velocities):
        """
        Sets velocities on the robot joints

        Parameters:
        -----------
            joints: the joints of the robot to move
                    given as a list of strings
            normalized_velocities: the velocities to apply
        """
        for joint, velocity in zip(joints, normalized_velocities):

            position = self.pepper.getAnglesPosition(joint)
            lower_limit = self.pepper.joint_dict[joint].getLowerLimit()
            upper_limit = self.pepper.joint_dict[joint].getUpperLimit()

            if position <= lower_limit and velocity < 0.0:
                velocity = 0.0
                self.episode_failed = True
            elif position >= upper_limit and velocity > 0.0:
                velocity = 0.0
                self.episode_failed = True
            # Decrease velocity for the head joints
            if joint == "HeadPitch" or joint == "HeadYaw":
                velocity *= self.HEAD_VELOCITY
            else:
                velocity *= self.limit_max * self.JOINTS_VELOCITY

            pybullet.setJointMotorControl2(
                self.pepper.robot_model,
                self.pepper.joint_dict[joint].getIndex(),
                pybullet.VELOCITY_CONTROL,
                targetVelocity=velocity,
                force=self.pepper.joint_dict[joint].getMaxEffort())

    def _getLinkPosition(self, link_name, robot):
        """
        Returns the position of the link in the world frame

        Parameters:
        -----------
            link_name: the name of the link given as a string
            robot: the robot instance to manipulate
        """
        link_state = pybullet.getLinkState(
            robot.robot_model,
            robot.link_dict[link_name].getIndex())

        return link_state[0], link_state[1]

    def normalize(self, values, range_min=-1.0, range_max=1.0):
        """
        Normalizes values (list) according to a specific range

        Parameters:
            values: the values to normalize
            range_min: the minimum value of the normalization
            range_max: the maximul value of the normalization
        Return:
            a list of the normalized values
        """
        zero_bound = [x - min(values) for x in values]
        range_bound = [
            x * (range_max - range_min) / (max(zero_bound) - min(zero_bound))
            for x in zero_bound]

        return [x - max(range_bound) + range_max for x in range_bound]

    def normalize_with_bounds(self, values, range_min, range_max,
                              bound_min, bound_max):
        """
        Normalizes values (list) according to a specific range given the bound
        limits
        Parameters:
            values: the values to be normalized
            range_min: the minimum value of the normalization
            range_max: the maximum value of the normalization
            bound_low: the low bound of the values to be normalized
            bound_high: the high bound of the values to be normalized
        """
        if isinstance(values, float):
            values = [values]
        values_std = [(x - bound_min) / (bound_max-bound_min)
                      for x in values]
        values_scaled = [x * (range_max - range_min) + range_min
                         for x in values_std]

        return values_scaled

    def _getState(self):
        """
        Gets the observation and computes the current reward. Will also
        determine if the episode is over or not, by filling the episode_over
        boolean.

        Returns:
            obs: the observations of the episode
            reward: the reward of the episode
        """
        reward = 0.0

        # Fill the observation
        obs = self._getObservation()

        # Get the last distance betwen the Hand and the Target
        hands_norm = self.hands_norm

        # Get the last distance between the unit orientation vector
        # of the Head in the world frame and the unit orientation vector
        # of the Head position and the Hand position
        orientations_norm = self.orientations_norm

        # Compute the reward of the Hand
        hand_reward = np.exp(-1 * hands_norm)

        # Compute the reward of the Head
        head_reward = np.exp(-1 * orientations_norm)

        # Compute total reward
        reward = (hand_reward * self.HAND_REWARD_WEIGHT +
                  head_reward * self.HEAD_REWARD_WEIGHT)

        # Check if there is a self collision on the r_wrist, if so stop the
        # episode. Else, check if the ball is below 0.049 (height of the
        # trash's floor)
        if self.pepper.isSelfColliding("r_wrist") or\
                self.pepper.isSelfColliding("RForeArm"):
            self.episode_over = True
            self.episode_failed = True
            i = 1
        # Check if the episode time has reached the allowed time
        elif(time.time() - self.episode_start_time) > self.EPISODE_TIME:
            self.episode_over = True
            self.episode_failed = True

        # Check desired limits for some joints
        elif(self.pepper.getAnglesPosition("HipPitch") < self.HIP_PITCH_MIN or
             self.pepper.getAnglesPosition("HipPitch") > self.HIP_PITCH_MAX):
            i = 1
            self.episode_over = True
            self.episode_failed = True

        self.episode_reward += reward

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

        # Get position of the Target (hand of the second pepper)
        # in the odom frame
        target,  target_rot =\
            self._getLinkPosition("l_gripper", self.pepper_bis)

        # Get position of the hand of the pepper  in the odom frame
        hand_pos,  _ = self._getLinkPosition("r_gripper", self.pepper)
        head_pos,  head_rot = self._getLinkPosition("Head", self.pepper)

        hand_pos = np.array(hand_pos)
        head_pos = np.array(head_pos)
        target = np.array(target)

        # Move the target on the y-axis because it is inside the hand
        target_bis = target
        target_bis[1] = target_bis[1] - 0.05

        hands_norm = np.linalg.norm(hand_pos - target_bis)

        # Get information about the head direction and hand
        # Head to Hand reward based on the direction of the head to the hand
        head_target_norm = np.linalg.norm(head_pos - target)
        head_target_vec = target - head_pos
        head_target_unit_vec = head_target_vec / head_target_norm

        # Create an objet for the rotation of the head
        head_rot_obj = R.from_quat([head_rot[2],
                                    head_rot[1],
                                    head_rot[0],
                                    head_rot[3]])

        # Apply the rotation on the z-axis
        rot_vec_ = head_rot_obj.apply([0, 0, 1])

        # Compute the unit direction vector ofthe head in the world frame
        head_norm = np.linalg.norm(rot_vec_)
        head_unit_vec = (rot_vec_ / head_norm)
        head_unit_vec = np.array([head_unit_vec[2],
                                  -head_unit_vec[1],
                                  -head_unit_vec[0]])

        # Compute de normal distance between both orientations
        orientations_norm = np.linalg.norm(
                np.array(head_target_unit_vec) - head_unit_vec)

        # Fill and return the observation
        hand_poses = [pose for pose in hand_pos]
        norm_hand_poses = self.normalize_with_bounds(
                          hand_poses, -1, 1, self.TARGET_MIN, self.TARGET_MAX)

        target_poses = [target_pose for target_pose in target]
        norm_target_poses = self.normalize_with_bounds(
                    target_poses, -1, 1, self.TARGET_MIN, self.TARGET_MAX)

        norm_joint_angles = list()
        for joint in self.r_kinematic_chain:

            if joint == "HipPitch":
                bound_max = self.HIP_PITCH_MAX
                bound_min = self.HIP_PITCH_MIN
            else:
                bound_max = self.pepper.joint_dict[joint].getUpperLimit()
                bound_min = self.pepper.joint_dict[joint].getLowerLimit()
            joint_angle = self.pepper.getAnglesPosition(joint)
            norm_joint_angle = self.normalize_with_bounds(
                            joint_angle, -1, 1, bound_min, bound_max
                         )
            norm_joint_angles.extend(norm_joint_angle)

        obs = [a for a in norm_joint_angles] +\
            [n for n in norm_hand_poses] +\
            [n for n in norm_target_poses] +\
            [e for e in head_target_unit_vec] +\
            [e for e in head_unit_vec]

        self.hands_norm = hands_norm
        self.orientations_norm = orientations_norm
        return obs

    def _setupScene(self):
        """
        Setup a scene environment within the simulation
        """
        self.client = self.simulation_manager.launchSimulation(gui=self.gui)
        self.pepper = self.simulation_manager.spawnPepper(
            self.client,
            spawn_ground_plane=True)

        self.pepper.goToPosture("Stand", 1.0)
        time.sleep(0.5)

        self.pepper_bis = self.simulation_manager.spawnPepper(
            self.client,
            translation=[0.80, -0.25, 0],
            quaternion=[0, 0, 1, 0],
            spawn_ground_plane=True)

        self.pepper_bis.goToPosture("Stand", 1.0)
        self.pepper_bis.setAngles("LShoulderPitch", 0.5, 1.0)

        time.sleep(0.5)
        self.joints_initial_pose = self.pepper.getAnglesPosition(
            self.pepper.joint_dict.keys())
        self.joints_initial_pose_bis = self.pepper_bis.getAnglesPosition(
            self.pepper.joint_dict.keys())

        pybullet.setAdditionalSearchPath("environment")

        time.sleep(0.2)

    def _resetScene(self):
        """
        Resets the scene for a new scenario
        """
        pybullet.resetBasePositionAndOrientation(
            self.pepper.robot_model,
            posObj=[0.0, 0.0, 0.0],
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client)

        self._hardResetJointState(self.pepper, self.joints_initial_pose)

        x = round(np.random.random(), 2)
        y = round(np.random.random(), 2)
        pybullet.resetBasePositionAndOrientation(
            self.pepper_bis.robot_model,
            posObj=[0.4 + (x*0.8), -0.5 + y, 0.0],
            ornObj=[0.0, 0.0, 1.0, 0.0],
            physicsClientId=self.client)
        r = round(np.random.random(), 2)
        self._hardResetJointState(self.pepper_bis,
                                  self.joints_initial_pose_bis)
        self.pepper_bis.setAngles("LShoulderPitch", -1 + r*2, 1.0)

        time.sleep(0.2)

    def _hardResetJointState(self, robot, initial_position):
        """
        Performs a hard reset on the joints of the robot, avoiding the robot to
        get stuck in a position
        """
        for joint, position in\
                zip(robot.joint_dict.keys(), initial_position):
            pybullet.setJointMotorControl2(
                robot.robot_model,
                robot.joint_dict[joint].getIndex(),
                pybullet.VELOCITY_CONTROL,
                targetVelocity=0.0)
            pybullet.resetJointState(
                    robot.robot_model,
                    robot.joint_dict[joint].getIndex(),
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
        "--train",
        type=int,
        default=1,
        help="True: training, False: using a trained model")

    parser.add_argument(
        "--algo",
        type=str,
        default="ppo2",
        help="The learning algorithm to be used (ppo2)")

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

    # Train a model
    if args.train == 1:
        print("train")
        if args.model != "":
            word = args.model
            path = "models/" + algo + "_hand_" + word
            print(args.model)
            log_name = "./logs/hand/" + word
            model = PPO2.load(path, vec_env, tensorboard_log=log_name)
        else:
            req = Request(
                "https://frightanic.com/goodies_content/docker-names.php",
                headers={'User-Agent': 'Mozilla/5.0'})

            webpage = str(urlopen(req).read())
            word = webpage.split("b\'")[1]
            word = word.split("\\")[0]
            word.replace(" ", "_")

            log_name = "./logs/hand/" + word

            if algo == "ppo2":
                policy_kwargs = dict(net_arch=[512, 512])
                model = PPO2(
                    stable_baselines.common.policies.MlpPolicy,
                    vec_env,
                    nminibatches=1,
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                    tensorboard_log=log_name)
            elif algo == "ddpg":
                model = DDPG(
                    CustomDDPGPolicy,
                    vec_env,
                    tensorboard_log=log_name)
            else:
                print("Please specify a valid algorithm using\
                    --algo (ppo, ddpg)")

        try:
            for episode in range(0, EPOCHS):
                model.learn(total_timesteps=EPISODES)
                model.save("models/" + algo + "_hand_" + word+"_"+str(episode))

        except KeyboardInterrupt:
            print("#---------------------------------#")
            print("Training \'" + word + "\' interrupted")
            print("#---------------------------------#")
            model.save("models/" + algo + "_hand_" + word)
            sys.exit(1)

        model.save("models/" + algo + "_hand_" + word)

    # Use a trained model
    else:
        if args.model == "":
            print("Specify the version of the model using --model")
            return

        path = "models/" + algo + "_hand_" + args.model
        if algo == "ppo2":
            model = PPO2.load(path)
        elif algo == "ddpg":
            model = DDPG.load(path)

        for test in range(100):
            dones = False
            obs = vec_env.reset()

            while not dones:
                action, _states = model.predict(
                    obs,
                    deterministic=DETERMINISTIC)
                obs, rewards, dones, info = vec_env.step(action)

    time.sleep(2)
    vec_env._termination()


if __name__ == "__main__":
    main()
