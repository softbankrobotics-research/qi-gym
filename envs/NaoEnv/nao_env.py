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

OBS_DIM = 224
DIST_MAX = 0.3
TIME_LIMIT = 5
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
            'l_ankle',
            'r_ankle',
            'LThigh',
            'r_sole',
            'RThigh',
            'LAnklePitch',
            'l_sole',
            'LTibia',
            'RTibia',
            'RHip',
            'RPelvis',
            'RAnklePitch',
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

        self.controlled_joints_reduced = [
            'RAnklePitch',
            'LHipRoll',
            'LKneePitch',
            'RHipRoll',
            'RHipPitch',
            'LHipYawPitch',
            'RHipYawPitch',
            'LHipPitch',
            'RAnkleRoll',
            'LAnkleRoll',
            'RKneePitch',
            'LAnklePitch']

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
            0.10,
            1.84,
            0.27,
            -1.50,
            -0.50,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.35,
            0.84,
            -0.50,
            0.0,
            0.0,
            0.0,
            -0.35,
            0.84,
            -0.5,
            0.0,
            1.84,
            -0.27,
            1.5,
            0.50,
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
        self.last_action = np.array([0] * len(self.controlled_joints_reduced))
        self.observation_space = spaces.Box(
            low=np.array([-3]*OBS_DIM),
            high=np.array([3]*OBS_DIM))
        self.turn = 0
        self.actual_traj = []
        self.action_space = spaces.Box(
            low=np.array([-1]*len(self.controlled_joints_reduced)),
            high=np.array([1]*len(self.controlled_joints_reduced)))
        self.frequency = H_STEPS

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
            assert len(action) == len(self.controlled_joints_reduced)

        except AssertionError:
            print("Incorrect action")
            return None, None, None, None
        np.clip(action, [-1]*len(self.controlled_joints_reduced),
                [1]*len(self.controlled_joints_reduced))
        self._setPositions(self.controlled_joints_reduced, action)
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
        self._resetScene()
        self.turn += 1
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
        actions = []
        for joint, n_position in zip(joints, n_positions):
            upper = self.nao.joint_dict[joint].getUpperLimit()
            lower = self.nao.joint_dict[joint].getLowerLimit()
            position = ((n_position + 1) * (upper - lower)) / 2 + lower
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
        # Get the information on the links
        (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw) =\
            self._getLinkState("torso")
        torso_state = np.array(
            [x, y, z, vx, vy, vz, qx, qy, qz, qw],
            dtype=np.float32
        )
        root_state = np.array(
            [z, qw, qx, qy, qz],
            dtype=np.float32
        )
        joint_pos = []
        joint_vel = []

        link_pos = []
        for link_name in self.link_list:
            (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw) =\
                self._getLinkState(link_name)
            link_pos += [
                x, y, z,
                qx, qy, qz, qw,
                vx, vy, vz,
                vroll, vpitch, vyaw]
        for name in self.controlled_joints_reduced:
            upper = self.nao.joint_dict[name].getUpperLimit()
            lower = self.nao.joint_dict[name].getLowerLimit()
            position = self.nao.getAnglesPosition(name)
            numerator = (position - lower)
            denominator = (upper - lower)
            joint_pos += [2 * (numerator / denominator) - 1]
            pos, vel, acc =\
                self._getJointState(name)
            joint_vel += [vel]
        joint_pos_state = np.array(
            joint_pos,
            dtype=np.float32
        )
        link_pos_state = np.array(
            link_pos,
            dtype=np.float32
        )
        joint_vel_state = np.array(
            joint_vel,
            dtype=np.float32
        )
        # joint_acc_state = np.array(
        #     joint_acc,
        #     dtype=np.float32
        # )
        counter = np.array(
            [self.counter / 200.0],
            dtype=np.float32
        )
        (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw) =\
            self._getLinkState("r_ankle")
        roll, pitch, yaw = pybullet.getEulerFromQuaternion([qx, qy, qz, qw])
        r_ankle_state = np.array(
            [x, y, z],
            dtype=np.float32
        )

        (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw) =\
            self._getLinkState("l_ankle")
        roll, pitch, yaw = pybullet.getEulerFromQuaternion([qx, qy, qz, qw])
        l_ankle_state = np.array(
            [x, y, z],
            dtype=np.float32
        )
        feet_pos = None
        feet_contact = np.array(
            self._getContactFeet(),
            dtype=np.float32
        )
        if r_ankle_state[0] >= l_ankle_state[0] and\
                feet_contact[0] == 0:
            feet_pos = "Right"
        if l_ankle_state[0] > r_ankle_state[0] and\
                feet_contact[1] == 0:
            feet_pos = "Left"
        if self.feet_ahead is None:
            self.feet_ahead = feet_pos
        if self.feet_ahead is not None and feet_pos is not None and\
                self.feet_ahead is not feet_pos:
            self.foot_step_number += 1
            self.feet_ahead = feet_pos
        # # Fill the observation
        obs = np.concatenate(
            [root_state] +
            [joint_pos_state] +
            [joint_vel_state] +
            [self.last_action] +
            [link_pos_state] +
            [counter])
        reward = 0
        # To be passed to True when the episode is over
        if torso_state[0] > DIST_MAX:
            self.episode_over = True
            reward += torso_state[0]
        if torso_state[2] < 0.27 or torso_state[0] < -1 or\
                time.time() - self.time_init > TIME_LIMIT:
            reward += torso_state[0]
            self.episode_over = True
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
        pybullet.resetBasePositionAndOrientation(
            self.nao.getRobotModel(),
            posObj=[0.0, 0.0, 0.36],
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
            childFramePosition=[0.0, 0.0, 0.36],
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
