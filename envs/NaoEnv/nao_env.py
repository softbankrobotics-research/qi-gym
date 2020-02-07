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

OBS_DIM = 48
DIST_MAX = 1.0
DISCOUNT_FACTOR = 0.1
H_STEPS = 10.0
SPEED_TO_ATTAIN = 0.0079
# SPEED_TO_ATTAIN = 0.05


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
        fix_value = 2
        obs_space_high = np.concatenate(
                [np.array(len(self.controlled_joints) * [math.pi])] +\
                [np.array(len(self.controlled_joints) * [fix_value])] +\
                [np.array([fix_value, fix_value, fix_value, math.pi,
                           fix_value])] +\
                [np.array([fix_value, fix_value, fix_value, math.pi])] +\
                [np.array([fix_value, fix_value, fix_value, math.pi])] +\
                [np.array([fix_value, fix_value, fix_value, math.pi])] +\
                [np.array([fix_value, fix_value, fix_value, math.pi])] +\
                [np.array([200.0])] +\
                [np.array([1.0, 1.0])]
                )
        obs_space_low = np.concatenate(
                [np.array(len(self.controlled_joints) * [-math.pi])] +\
                [np.array(len(self.controlled_joints) * [-fix_value])] +\
                [np.array([-fix_value, -fix_value, -fix_value, -math.pi,
                           -fix_value])] +\
                [np.array([-fix_value, -fix_value, -fix_value, -math.pi])] +\
                [np.array([-fix_value, -fix_value, -fix_value, -math.pi])] +\
                [np.array([-fix_value, -fix_value, -fix_value, -math.pi])] +\
                [np.array([-fix_value, -fix_value, -fix_value, -math.pi])] +\
                [np.array([0.0])] +\
                [np.array([0.0, 0.0])]
                )

        self.observation_space = spaces.Box(
            low=obs_space_low,
            high=obs_space_high)

        self.action_space = spaces.Box(
            low=np.array([0]*len(self.controlled_joints)),
            high=np.array([1]*len(self.controlled_joints)))
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
        self._resetScene()

        obs, _ = self._getState()
        # if self.plot_list:
        #     import matplotlib.pyplot as plt
        #     plt.plot(self.plot_list)
        #     plt.show()
        # self.plot_list = []
        return obs

    def _hardResetJointState(self):
        for joint, position in zip(self.all_joints, self.starting_position):
            pybullet.setJointMotorControl2(
                self.nao.robot_model,
                self.nao.joint_dict[joint].getIndex(),
                pybullet.VELOCITY_CONTROL,
                targetVelocity=0,
                force=self.nao.joint_dict[joint].getMaxEffort(),
                physicsClientId=self.client)
            pybullet.resetJointState(
                    self.nao.robot_model,
                    self.nao.joint_dict[joint].getIndex(),
                    position,
                    physicsClientId=self.client)
        self._resetJointState()

    def _resetJointState(self):
        self.nao.setAngles(self.all_joints,
                           self.starting_position, 1.0)

    def render(self, mode='human', close=False):
        pass

    def _setVelocities(self, joints, n_velocities):
        """
        Sets velocities on the robot joints
        """

        for joint, n_velocity in zip(joints, n_velocities):
            velocity = n_velocity * self.nao.joint_dict[joint].getMaxVelocity()
            pybullet.setJointMotorControl2(
                self.nao.robot_model,
                self.nao.joint_dict[joint].getIndex(),
                pybullet.VELOCITY_CONTROL,
                targetVelocity=velocity,
                force=self.nao.joint_dict[joint].getMaxEffort(),
                physicsClientId=self.client)

    def _setPositions(self, joints, n_positions):
        """
        Sets positions on the robot joints
        """

        for joint, n_position in zip(joints, n_positions):
            upper = self.nao.joint_dict[joint].getUpperLimit()
            lower = self.nao.joint_dict[joint].getLowerLimit()
            position = n_position * (upper - lower) + lower
            pybullet.setJointMotorControl2(
                self.nao.robot_model,
                self.nao.joint_dict[joint].getIndex(),
                pybullet.POSITION_CONTROL,
                targetPosition=position,
                force=self.nao.joint_dict[joint].getMaxEffort(),
                physicsClientId=self.client)

    def _getJointState(self, joint_name):
        """
        Returns the state of the joint in the world frame
        """
        position, velocity, _, _ =\
            pybullet.getJointState(
                self.nao.robot_model,
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
            self.nao.robot_model,
            self.nao.link_dict[link_name].getIndex(),
            computeLinkVelocity=1,
            physicsClientId=self.client)

        return (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw)

    def _getContactFeet(self):
        foot_list = ["r_ankle", "l_ankle"]
        contact_list = []
        for foot_joint in foot_list:
            points = pybullet.getContactPoints(
                bodyA=self.nao.robot_model,
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
            bodyA=self.nao.robot_model,
            linkIndexA=self.nao.link_dict[foot_list[0]].getIndex(),
            bodyB=self.nao.robot_model,
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
                bodyA=self.nao.robot_model,
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

        # Get the information on the joints
        joint_position_list = []
        joint_velocity_list = []
        joint_acceleration_list = []

        for joint in self.controlled_joints:
            pos, vel, acc = self._getJointState(joint)
            joint_position_list.append(pos)
            joint_velocity_list.append(vel)
            joint_acceleration_list.append(acc)

        joint_position_list = np.array(
            joint_position_list,
            dtype=np.float32
        )

        joint_velocity_list = np.array(
            joint_velocity_list,
            dtype=np.float32
        )

        joint_acceleration_list = np.array(
            joint_acceleration_list,
            dtype=np.float32
        )

        feet_contact = np.array(
            self._getContactFeet(),
            dtype=np.float32
        )

        (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw) =\
            self._getLinkState("torso")
        _, _, yaw = pybullet.getEulerFromQuaternion([qx, qy, qz, qw])
        torso_state = np.array(
            # [x, y, z, yaw, vx, vy, vz, vroll, vpitch, vyaw],
            [x, y, z, yaw, vx],
            dtype=np.float32
        )

        (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw) =\
            self._getLinkState("LHip")
        _, _, yaw = pybullet.getEulerFromQuaternion([qx, qy, qz, qw])
        l_hip_state = np.array(
            [x, y, z, yaw],
            dtype=np.float32
        )
        (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw) =\
            self._getLinkState("RHip")
        _, _, yaw = pybullet.getEulerFromQuaternion([qx, qy, qz, qw])
        r_hip_state = np.array(
            [x, y, z, yaw],
            dtype=np.float32
        )
        (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw) =\
            self._getLinkState("l_ankle")
        _, _, yaw = pybullet.getEulerFromQuaternion([qx, qy, qz, qw])
        l_ankle_state = np.array(
            [x, y, z, yaw],
            dtype=np.float32
        )
        (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw) =\
            self._getLinkState("r_ankle")
        _, _, yaw = pybullet.getEulerFromQuaternion([qx, qy, qz, qw])
        r_ankle_state = np.array(
            [x, y, z, yaw],
            dtype=np.float32
        )

        counter = np.array(
            [self.counter],
            dtype=np.float32
        )
        # Fill the observation
        obs = np.concatenate(
            [joint_position_list] +
            [joint_velocity_list] +
            [torso_state] +
            [l_hip_state] +
            [r_hip_state] +
            [l_ankle_state] +
            [r_ankle_state] +
            [counter] +
            [feet_contact])
        # obs = np.concatenate(
        #     [counter] +
        #     [joint_position_list] +
        #     [joint_velocity_list] +
        #     [joint_acceleration_list] +
        #     [torso_state] +
        #     [feet_contact])
        reward = 0
        # To be passed to True when the episode is over
        feet_pose = abs(l_ankle_state[0]-r_ankle_state[0])/2 +\
                np.min([l_ankle_state[0] , r_ankle_state[0]])
        if torso_state[0] > DIST_MAX:
            reward += 1
            reward += (np.log((min([l_ankle_state[0],
                                   r_ankle_state[0],
                                   r_hip_state[0],
                                   l_hip_state[0],
                                   torso_state[0]
                                   ]
                                   ) + 1)) / 1.4)
            reward -= time.time() - self.time_init
            self.episode_over = True
        if torso_state[2] < 0.27 or torso_state[0] < -1 or\
                time.time() - self.time_init > 20:
            reward += -1
            reward += (np.log((min([feet_pose,
                                   r_hip_state[0],
                                   l_hip_state[0],
                                   torso_state[0]
                                   ]
                                   ) + 1)) / 1.4)
            self.episode_over = True
        # Compute the reward
        # delta x : speed
        # reward += torso_state[0] - self.previous_x

        # if (self.counter != 0):
        #     self.plot_list.append(a - self.previous_x)
        reward += np.exp(-10000*((feet_pose - self.previous_x) - 
                            SPEED_TO_ATTAIN)**2) / 400.0
        if 1 in self._getContactHands():
            reward += -0.003
        # self.previous_x = torso_state[0]
        self.previous_x = feet_pose
        return obs, reward

    def _setupScene(self):
        """
        Setup a scene environment within the simulation
        """
        self.client = self.simulation_manager.launchSimulation(gui=self.gui)
        self.nao = self.simulation_manager.spawnNao(
            self.client,
            spawn_ground_plane=True)
        # self.nao.goToPosture("Stand", 0.8)

        # time.sleep(1.5)

        # self.starting_position = self.nao.getAnglesPosition(self.all_joints)
        self._resetJointState()
        time.sleep(1.0)

    def _resetScene(self):
        """
        Resets the scene for a new scenario
        """

        pybullet.resetBasePositionAndOrientation(
            self.nao.robot_model,
            posObj=[0.0, 0.0, 0.36],
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client)
        balance_constraint = pybullet.createConstraint(
            parentBodyUniqueId=self.nao.robot_model,
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
            balance_constraint,
            physicsClientId=self.client)
        time.sleep(0.5)

    def _termination(self):
        """
        Terminates the environment
        """
        self.simulation_manager.stopSimulation(self.client)

    # def close(self):
    #     self._termination()
    
