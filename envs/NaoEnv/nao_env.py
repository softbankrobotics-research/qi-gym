#!/usr/bin/env python
# coding: utf-8
import sys
CV2_ROS = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if CV2_ROS in sys.path:
    sys.path.remove(CV2_ROS)
    sys.path.append(CV2_ROS)
import gym
import time
import numpy as np
from gym import spaces
import pybullet
from qibullet import SimulationManager
import math
import json
from scipy.spatial.transform import Rotation as R
import random


OBS_DIM = 101
DIST_MAX = 1.0
H_STEPS = 10.0


class NaoEnv(gym.Env):
    """
    Gym environment for the NAO robot, walking task
    """

    def __init__(self, gui=False):
        """
        Constructor

        Parameters:
            gui - boolean, the simulation is in DIRECT mode if set to False
            (default value)
        """
        self.link_list = [
            'Head',
            'l_ankle',
            'LForeArm',
            'RForeArm',
            'r_ankle',
            'torso',
            'LThigh',
            'RBicep',
            'r_sole',
            'RThigh',
            'LAnklePitch',
            'LBicep',
            'l_sole',
            'LTibia',
            'r_wrist',
            'LElbow',
            'RElbow',
            'RTibia',
            'Neck',
            'l_wrist',
            'RHip',
            'RPelvis',
            'LShoulder',
            'RAnklePitch',
            'RShoulder',
            'LHip',
            'LPelvis'
        ]
        self.controlled_joints = [
            'RAnklePitch',
            'LHipRoll',
            'LKneePitch',
            'RShoulderPitch',
            'RHipRoll',
            'RHipPitch',
            'LHipYawPitch',
            'RShoulderRoll',
            'RHipYawPitch',
            'LElbowYaw',
            'LHipPitch',
            'RAnkleRoll',
            'LAnkleRoll',
            'LShoulderRoll',
            'RKneePitch',
            'LElbowRoll',
            'RElbowYaw',
            'RElbowRoll',
            'LAnklePitch',
            'LShoulderPitch']

        self.all_joints = [
            "HeadYaw",
            "HeadPitch",
            "LShoulderPitch",
            "LShoulderRoll",
            "LElbowYaw",
            "LElbowRoll",
            "LWristYaw",
            "LHand",
            "LHipYawPitch",
            "LHipRoll",
            "LHipPitch",
            "LKneePitch",
            "LAnklePitch",
            "LAnkleRoll",
            "RHipYawPitch",
            "RHipRoll",
            "RHipPitch",
            "RKneePitch",
            "RAnklePitch",
            "RAnkleRoll",
            "RShoulderPitch",
            "RShoulderRoll",
            "RElbowYaw",
            "RElbowRoll",
            "RWristYaw",
            "RHand"]

        self.starting_position = [
            0.0,
            0.0,
            1.57079632,
            0.0,
            -1.57079632,
            -1.57079632,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.523598775,
            1.04719755,
            -0.523598775,
            0.0,
            0.0,
            0.0,
            -0.523598775,
            1.04719755,
            -0.523598775,
            0.0,
            1.570796326,
            0.0,
            1.570796326,
            1.570796326,
            0.0,
            0.0]

        # Passed to True at the end of an episode
        self.episode_over = False
        self.gui = gui
        self.simulation_manager = SimulationManager()
        self.counter = 0
        self.foot_step_number = 0
        self.feet_ahead = None
        self.plot_list = []
        self._setupScene()
        self.last_action = np.array([0] * len(self.controlled_joints))
        self.observation_space = spaces.Box(
            low=np.array([-3]*OBS_DIM),
            high=np.array([3]*OBS_DIM))

        self.action_space = spaces.Box(
            low=np.array([0]*len(self.controlled_joints)),
            high=np.array([1]*len(self.controlled_joints)))
        self.frequency = H_STEPS
        with open('data/nao/dataset_walk_nao_saved.json') as json_file:
            self.expert_dataset = json.load(json_file)
        self.expert_joint_pos = self.expert_dataset["joint_position"].copy()
        self.expert_link_pos = self.expert_dataset["link_position"].copy()
        self.expert_link_vel = self.expert_dataset["link_velocity"].copy()
        self.expert_torso_vel = self.expert_dataset["torso_velocity"].copy()
        self.expert_link_ori = self.expert_dataset["link_orientation"].copy()
        self.expert_head_pos = self.expert_dataset["head_position"].copy()

    def step(self, action):
        """

        Parameters
        ----------
        action : list of velocities to be applied on the robot's joints

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
            assert len(action) == len(self.controlled_joints)

        except AssertionError:
            print("Incorrect action")
            return None, None, None, None
        np.clip(action, [0]*len(self.controlled_joints),
                [1]*len(self.controlled_joints))
        self._setPositions(self.controlled_joints, action)
        self.counter += 1
        time.sleep(1.0 / self.frequency)
        obs, reward = self._getState()
        return obs, reward, self.episode_over, {}

    def reset(self):
        """
        Resets the environment for a new episode
        """
        self.episode_over = False
        self.previous_x = 0
        self.previous_vel = 0
        self.previous_time = time.time()
        self.counter = 0
        self.foot_step_number = 0
        self.time_init = time.time()
        self.feet_ahead = None
        self.expert_joint_pos = self.expert_dataset["joint_position"].copy()
        self.expert_link_pos = self.expert_dataset["link_position"].copy()
        self.expert_link_vel = self.expert_dataset["link_velocity"].copy()
        self.expert_torso_vel = self.expert_dataset["torso_velocity"].copy()
        self.expert_link_ori = self.expert_dataset["link_orientation"].copy()
        self.expert_head_pos = self.expert_dataset["head_position"].copy()
        random_number = int(random.random() * 50)
        self.expert_joint_pos = self.expert_joint_pos[random_number:]
        self.expert_link_pos = self.expert_link_pos[random_number:]
        self.expert_link_vel = self.expert_link_vel[random_number:]
        self.expert_torso_vel = self.expert_torso_vel[random_number:]
        self.expert_link_ori = self.expert_link_ori[random_number:]
        self.expert_head_pos = self.expert_head_pos[random_number:]
        self.starting_position = self.expert_joint_pos[0]
        self.all_joints = self.controlled_joints.copy()
        self._resetScene()

        obs, _ = self._getState()

        return obs

    def _hardResetJointState(self):
        for joint, position in zip(self.all_joints, self.starting_position):
            pybullet.setJointMotorControl2(
                self.nao.getRobotModel(),
                self.nao.joint_dict[joint].getIndex(),
                pybullet.VELOCITY_CONTROL,
                targetVelocity=0,
                force=self.nao.joint_dict[joint].getMaxEffort(),
                physicsClientId=self.client)
            pybullet.resetJointState(
                    self.nao.getRobotModel(),
                    self.nao.joint_dict[joint].getIndex(),
                    position,
                    physicsClientId=self.client)
        self._resetJointState()

    def _resetJointState(self):
        self.nao.setAngles(self.all_joints,
                           self.starting_position, 1.0)

    def render(self, mode='human', close=False):
        pass

    def _setPositions(self, joints, n_positions):
        """
        Sets positions on the robot joints
        """
        if len(self.expert_link_pos) <= 2 or\
                len(self.expert_joint_pos) <= 2 or\
                len(self.expert_link_vel) <= 2 or\
                len(self.expert_torso_vel) <= 2 or\
                len(self.expert_link_ori) <= 2 or\
                len(self.expert_head_pos) <= 2:
            self.episode_over = True
            return
        actions = []
        for joint, n_position in zip(joints, n_positions):
            upper = self.nao.joint_dict[joint].getUpperLimit()
            lower = self.nao.joint_dict[joint].getLowerLimit()
            position = n_position * (upper - lower) + lower
            pybullet.setJointMotorControl2(
                self.nao.getRobotModel(),
                self.nao.joint_dict[joint].getIndex(),
                pybullet.POSITION_CONTROL,
                targetPosition=position,
                force=self.nao.joint_dict[joint].getMaxEffort(),
                physicsClientId=self.client)
            actions.append(position)
        self.last_action = np.array(
            actions
        )

    def _getJointState(self, joint_name):
        """
        Returns the state of the joint in the world frame
        """
        position, velocity, _, _ =\
            pybullet.getJointState(
                self.nao.getRobotModel(),
                self.nao.joint_dict[joint_name].getIndex(),
                physicsClientId=self.client)
        delta_time = time.time() - self.previous_time
        acceleration = (velocity - self.previous_vel) / delta_time
        self.previous_vel = velocity
        return position, velocity, acceleration

    def _getLinkState(self, link_name):
        """
        Returns the state of the link in the world frame
        """
        (x, y, z), (qx, qy, qz, qw), _, _, _, _, (vx, vy, vz),\
            (vroll, vpitch, vyaw) = pybullet.getLinkState(
            self.nao.getRobotModel(),
            self.nao.link_dict[link_name].getIndex(),
            computeLinkVelocity=1,
            physicsClientId=self.client)

        return (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw)

    def _getContactFeet(self):
        foot_list = ["r_ankle", "l_ankle"]
        contact_list = []
        for foot_joint in foot_list:
            points = pybullet.getContactPoints(
                bodyA=self.nao.getRobotModel(),
                linkIndexA=self.nao.link_dict[foot_joint].getIndex(),
                physicsClientId=self.client)
            if len(points) > 0:
                contact_list.append(1)
            else:
                contact_list.append(0)
        return contact_list

    def _getContactFootToFoot(self):
        foot_list = ["r_ankle", "l_ankle"]
        contact_list = []
        points = pybullet.getContactPoints(
            bodyA=self.nao.getRobotModel(),
            linkIndexA=self.nao.link_dict[foot_list[0]].getIndex(),
            bodyB=self.nao.getRobotModel(),
            linkIndexB=self.nao.link_dict[foot_list[1]].getIndex(),
            physicsClientId=self.client)
        if len(points) > 0:
            contact_list.append(1)
        else:
            contact_list.append(0)
        return contact_list

    def _getContactHands(self):
        hand_list = ["r_wrist", "l_wrist"]
        contact_list = []
        for hand_joint in hand_list:
            points = pybullet.getContactPoints(
                bodyA=self.nao.getRobotModel(),
                linkIndexA=self.nao.link_dict[hand_joint].getIndex(),
                physicsClientId=self.client)
            if len(points) > 0:
                contact_list.append(1)
            else:
                contact_list.append(0)
        return contact_list

    def _getState(self, convergence_criteria=0.12, divergence_criteria=0.6):
        """
        Gets the observation and computes the current reward. Will also
        determine if the episode is over or not, by filling the episode_over
        boolean. When the euclidian distance between the wrist link and the
        cube is inferior to the one defined by the convergence criteria, the
        episode is stopped
        """
        if len(self.expert_link_pos) <= 2 or\
                len(self.expert_joint_pos) <= 2 or\
                len(self.expert_link_vel) <= 2 or\
                len(self.expert_torso_vel) <= 2 or\
                len(self.expert_link_ori) <= 2 or\
                len(self.expert_head_pos) <= 2:
            self.episode_over = True
        # Get the information on the links
        (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw) =\
            self._getLinkState("torso")
        torso_pose = [x, y, z]
        torso_vel = np.array(
            [vx, vy, vz],
            dtype=np.float32
        )
        (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw) =\
            self._getLinkState("Head")
        head_state = np.array(
            [x, y, z],
            dtype=np.float32
        )
        link_pos = []
        link_vel = []
        link_ori = []
        for name in self.link_list:
            (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw) =\
                self._getLinkState(name)
            link_pos += [x, y, z]
            link_vel += [vx, vy, vz]
            link_ori += [qx, qy, qz, qw]
        link_pos_state = np.array(
            link_pos,
            dtype=np.float32
        )
        link_vel_state = np.array(
            link_vel,
            dtype=np.float32
        )
        link_ori_state = np.array(
            link_ori,
            dtype=np.float32
        )
        expert_link_pos = np.array(self.expert_link_pos[0])
        expert_link_vel = np.array(self.expert_link_vel[0])
        expert_torso_vel = np.array(self.expert_torso_vel[0])
        expert_link_ori = np.array(self.expert_link_ori[0])
        expert_head_pos = np.array(self.expert_head_pos[0])
        # # Fill the observation
        obs = np.concatenate(
            [expert_torso_vel] +
            [torso_vel] +
            [torso_vel - expert_torso_vel] +
            [link_pos_state[:6*3]] +
            [link_vel_state[:6*3]] +
            [link_pos_state[:6*3] - expert_link_pos[:6*3]] +
            [link_vel_state[:6*3] - expert_link_vel[:6*3]] +
            [self.last_action])

        reward = 0
        # To be passed to True when the episode is over
        if torso_pose[0] > DIST_MAX:
            self.episode_over = True

        rwd_torso = np.exp(-np.linalg.norm(
            expert_torso_vel - torso_vel))

        norm_sum = 0
        for i in range(0, len(expert_link_pos), 3):
            norm_sum +=\
                np.linalg.norm(
                    expert_link_pos[i:i+3] - link_pos_state[i:i+3])
        rwd_pos = np.exp(-10/len(self.link_list) * norm_sum)

        norm_sum = 0
        for i in range(0, len(expert_link_vel), 3):
            norm_sum +=\
                np.linalg.norm(
                    expert_link_vel[i:i+3] - link_vel_state[i:i+3])
        rwd_vel = np.exp(-1/len(self.link_list) * norm_sum)

        norm_sum = 0
        for i in range(0, len(expert_link_ori), 4):
            inv_quater = R.from_quat(link_ori_state[i:i+4])
            inv_quater = inv_quater.inv().as_quat()
            norm_sum +=\
                np.dot(expert_link_ori[i:i+4], inv_quater)
        rwd_local = np.exp(-10/len(self.link_list) * norm_sum)

        rwd_fall = max(1.3 - 1.4 *
                       np.linalg.norm(
                           expert_head_pos - head_state), 0.1)

        reward += rwd_fall * (rwd_pos + rwd_vel + rwd_local + rwd_torso)
        if torso_pose[2] < 0.27 or torso_pose[0] < -1 or\
                time.time() - self.time_init > 20:
            self.episode_over = True
        self.expert_link_pos.pop(0)
        self.expert_link_vel.pop(0)
        self.expert_torso_vel.pop(0)
        self.expert_link_ori.pop(0)
        self.expert_head_pos.pop(0)
        return obs, reward

    def _setupScene(self):
        """
        Setup a scene environment within the simulation
        """
        self.client = self.simulation_manager.launchSimulation(gui=self.gui)
        self.nao = self.simulation_manager.spawnNao(
            self.client,
            spawn_ground_plane=True)

        self._resetJointState()
        time.sleep(1.0)

    def _resetScene(self):
        """
        Resets the scene for a new scenario
        """
        torso_pos = self.expert_link_pos[0]
        torso_pos = torso_pos[3*5:3*5+3]
        pybullet.resetBasePositionAndOrientation(
            self.nao.getRobotModel(),
            posObj=[0.0 + torso_pos[0], 0.0 + torso_pos[1], 0.36],
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client)

        balance_constraint_nao = pybullet.createConstraint(
            parentBodyUniqueId=self.nao.getRobotModel(),
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=pybullet.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFramePosition=[0.0 + torso_pos[0], 0.0 + torso_pos[1], 0.36],
            childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client)
        self._hardResetJointState()
        pybullet.removeConstraint(
            balance_constraint_nao,
            physicsClientId=self.client)
        time.sleep(0.5)

    def _termination(self):
        """
        Terminates the environment
        """
        self.simulation_manager.stopSimulation(self.client)
