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
from numpy import linalg as LA
from gym import spaces
import pybullet
import pybullet_data
from qibullet import SimulationManager
from threading import Thread
import json
import random

OBS_DIM = 111
DIST_MAX = 1.0
TIME_LIMIT = 60
H_STEPS = 10.0

max_list_norm = [
    -0.1839316139255156, 1, 1, 0.9460295971137895, 1,
    1, 1, 0.3027593352252853, 1, -0.7147751666935444, 1,
    0.6138682889505842, -0.052568003024373144, -0.28340587935829, 1,
    0.3783612945820465, 0.7213998737207266, -0.3429896057857109,
    -0.23830955661457842, 0.9039678944301639]
min_list_norm = [
    -0.6290612055966212, -1, -1, 0.8559511285443029, -1,
    -1, -1, 0.28396991509754166, -1, -0.7264913615942559,
    0.09099978374453599, -0.5712173224970232, -1, -0.3000643888120029, -1,
    0.2925194447343942, 0.7099142475456754, -0.42662416484552057,
    -0.654380835225721, 0.8140488349275754]


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
        self.link_list_reduced = [
            'l_ankle',
            'LForeArm',
            'RForeArm',
            'r_ankle',
            'torso',
            'LThigh',
            'RBicep',
            'RThigh',
            'LBicep',
            'LTibia',
            'RTibia',
            'RPelvis',
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
        self.data_expert = None
        with open("data/nao/trajectories_walk.json") as json_file:
            self.data_expert = json.load(json_file)

        self.trajectories_expert = self.data_expert["trajectories"].copy()
        self.trajectories_expert[0] = self.trajectories_expert[0][12:21]
        self.starting_position = self.trajectories_expert[0][0]
        self.len_phase = float(len(self.trajectories_expert[0]))

        self.simulation_manager = SimulationManager()
        self.phase = 0.0
        self._setupScene()
        self.observation_space = spaces.Box(
            low=np.array([-1]*OBS_DIM),
            high=np.array([1]*OBS_DIM))
        self.action_space = spaces.Box(
            low=np.array(min_list_norm),
            high=np.array(max_list_norm))
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
        np.clip(action, min_list_norm, max_list_norm)
        if len(self.action_exp) == 0:
            self.action_exp = self.trajectories_expert[0].copy()
            self.phase = 0.0
        action_exp = self.action_exp[0]
        self._setPositions(
                [self.nao, self.nao_expert],
                self.controlled_joints,
                [action, action_exp])
        self.action_exp.pop(0)
        time.sleep(1.0 / self.frequency)
        obs, reward = self._getState()
        return obs, reward, self.episode_over, {}

    def reset(self):
        """
        Resets the environment for a new episode
        """
        self.episode_over = False
        self.phase = 0.0
        self.time_init = time.time()
        self.action_exp = self.trajectories_expert[0].copy()
        self.phase = int(random.random() * self.len_phase)
        self.action_exp = self.action_exp[self.phase:]
        self.starting_position = self.action_exp[0]
        self._resetScene()
        obs, _ = self._getState()
        return obs

    def _hardResetJointState(self, virtual_robot):
        for joint, position in zip(
                self.controlled_joints,
                self.starting_position):
            pybullet.setJointMotorControl2(
                virtual_robot.getRobotModel(),
                virtual_robot.joint_dict[joint].getIndex(),
                pybullet.VELOCITY_CONTROL,
                targetVelocity=0,
                force=virtual_robot.joint_dict[joint].getMaxEffort(),
                physicsClientId=self.client)
            pybullet.resetJointState(
                    virtual_robot.getRobotModel(),
                    virtual_robot.joint_dict[joint].getIndex(),
                    position,
                    physicsClientId=self.client)
        self._resetJointState(virtual_robot)

    def _resetJointState(self, virtual_robot):
        virtual_robot.setAngles(
            self.controlled_joints,
            self.starting_position, 1.0)

    def render(self, mode='human', close=False):
        pass

    def _setPositions(self, virtual_robot_list, joints, n_positions_list):
        """
        Sets positions on the robot joints
        """

        actions = []
        index = self.index_joint_list.copy()
        joint_force = self.joint_force_list.copy()
        upper = self.upper_joint_list.copy()
        lower = self.lower_joint_list.copy()
        cpt = 0
        n_positions_sim = n_positions_list[0]
        for joint, n_position in zip(joints, n_positions_sim):
            position = ((n_position + 1) * (upper[cpt] - lower[cpt]))\
                / 2 + lower[cpt]
            actions.append(position)
            cpt += 1
        actions_list = [actions, n_positions_list[1]]
        for virtual_robot, actions in zip(virtual_robot_list, actions_list):
            pybullet.setJointMotorControlArray(
                    virtual_robot.getRobotModel(),
                    index,
                    pybullet.POSITION_CONTROL,
                    targetPositions=actions,
                    forces=joint_force,
                    physicsClientId=self.client)

    def _getJointState(self, robot_virtual, joint_name):
        """
        Returns the state of the joint in the world frame
        """
        position, velocity, _, _ =\
            pybullet.getJointState(
                robot_virtual.getRobotModel(),
                robot_virtual.joint_dict[joint_name].getIndex(),
                physicsClientId=self.client)
        return position, velocity

    def _getLinkState(self, robot_virtual, link_name):
        """
        Returns the state of the link in the world frame
        """
        (x, y, z), (qx, qy, qz, qw), _, _, _, _, (vx, vy, vz),\
            (vroll, vpitch, vyaw) = pybullet.getLinkState(
            robot_virtual.getRobotModel(),
            robot_virtual.link_dict[link_name].getIndex(),
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
        pose_torso_sim, quater_torso_sim, _, _ =\
            self._getLinkState(self.nao, "torso")
        pose_torso_expert, quater_torso_expert, _, _ =\
            self._getLinkState(self.nao_expert, "torso")
        euler_torso_sim = pybullet.getEulerFromQuaternion(
            quater_torso_sim
        )
        euler_torso_expert = pybullet.getEulerFromQuaternion(
            quater_torso_expert
        )
        com_err = LA.norm(
            np.array(pose_torso_sim) - np.array(pose_torso_expert), 2)
        pose_err = 0
        vel_err = 0
        obs = np.concatenate(
            [np.array([self.phase / self.len_phase])] +
            [np.array(pose_torso_sim)] +
            [np.array(pose_torso_expert)])
        self.phase += 1
        for name in self.controlled_joints:
            pose_sim, vel_sim =\
                self._getJointState(self.nao, name)
            pose_sim = np.array([pose_sim])
            vel_sim = np.array([vel_sim])
            pose_exp, vel_exp =\
                self._getJointState(self.nao_expert, name)
            pose_exp = np.array([pose_exp])
            vel_exp = np.array([vel_exp])
            obs = np.concatenate(
                [obs] +
                [pose_sim] +
                [vel_sim] +
                [pose_exp] +
                [vel_exp])
            pose_err += LA.norm(
                    pose_sim - pose_exp, 2)
            vel_err += LA.norm(
                    vel_sim - vel_exp, 2)

        end_eff_err = 0
        for link_name in self.link_list_reduced:
            if link_name in ["r_ankle", "l_ankle", "LForeArm", "RForeArm"]:
                pose_sim, quater_sim, vlin_sim, vang_sim =\
                    self._getLinkState(self.nao, link_name)
                pose_sim = np.array(pose_sim)
                pose_exp, quater_exp, vlin_exp, vang_exp =\
                    self._getLinkState(self.nao_expert, link_name)
                pose_exp = np.array(pose_exp)
                end_eff_err += LA.norm(
                    pose_sim - pose_exp, 2)
                obs = np.concatenate(
                    [obs] +
                    [pose_sim] +
                    [pose_exp])
        reward = 0
        # To be passed to True when the episode is ove
        if pose_torso_sim[0] > DIST_MAX or pose_torso_expert[0] > DIST_MAX:
            self.episode_over = True
        if pose_torso_sim[2] < 0.27 or pose_torso_sim[0] < -1 or\
                pose_torso_expert[2] < 0.27 or\
                abs(pose_torso_expert[0] - pose_torso_sim[0]) >= 0.2 or\
                abs(euler_torso_expert[2] - euler_torso_sim[2]) >= 0.2:
            self.episode_over = True

        pose_reward = np.exp(-2 * pose_err)
        vel_reward = np.exp(-0.1 * vel_err)
        end_eff_reward = np.exp(-40 * end_eff_err)
        com_reward = np.exp(-10 * com_err)
        pose_w = 0.80
        vel_w = 0.05
        end_eff_w = 0.10
        com_w = 0.05
        reward_i = pose_w * pose_reward + vel_w * vel_reward +\
            end_eff_w * end_eff_reward + com_w * com_reward
        reward_g = 0
        reward = reward_i + reward_g
        return obs, reward

    def _setupScene(self):
        """
        Setup a scene environment within the simulation
        """
        self.client = self.simulation_manager.launchSimulation(gui=self.gui)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = pybullet.loadMJCF(
            "mjcf/ground_plane.xml",
            physicsClientId=self.client)[0]
        self.nao = self.simulation_manager.spawnNao(
            self.client)
        self.nao_expert = self.simulation_manager.spawnNao(
            self.client, translation=[0, 1, 0])
        pybullet.setCollisionFilterGroupMask(
            self.nao_expert.getRobotModel(),
            -1,
            collisionFilterGroup=0,
            collisionFilterMask=0)
        pybullet.setCollisionFilterPair(
            self.planeId, self.nao_expert.getRobotModel(), -1, -1, 1)
        alpha = 0.4
        pybullet.changeVisualShape(
            self.nao_expert.getRobotModel(), -1, rgbaColor=[1, 1, 1, alpha])
        for j in range(pybullet.getNumJoints(self.nao_expert.getRobotModel())):
            pybullet.setCollisionFilterGroupMask(
                self.nao_expert.getRobotModel(),
                j,
                collisionFilterGroup=0,
                collisionFilterMask=0)
            pybullet.setCollisionFilterPair(
                self.planeId, self.nao_expert.getRobotModel(), -1, j, 1)
            pybullet.changeVisualShape(
                self.nao_expert.getRobotModel(), j,
                rgbaColor=[1, 1, 1, alpha])

        self._resetJointState(self.nao)
        self._resetJointState(self.nao_expert)

        self.index_joint_list = []
        self.joint_force_list = []
        self.upper_joint_list = []
        self.lower_joint_list = []
        for joint in self.controlled_joints:
            self.index_joint_list.append(self.nao.joint_dict[joint].getIndex())
            self.joint_force_list.append(
                self.nao.joint_dict[joint].getMaxEffort())
            self.upper_joint_list.append(
                self.nao.joint_dict[joint].getUpperLimit())
            self.lower_joint_list.append(
                self.nao.joint_dict[joint].getLowerLimit())
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

        self.balance_constraint_nao = pybullet.createConstraint(
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

        pybullet.resetBasePositionAndOrientation(
            self.nao_expert.getRobotModel(),
            posObj=[0.0, 0.0, 0.36],
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client)

        self.balance_constraint_nao_expert = pybullet.createConstraint(
            parentBodyUniqueId=self.nao_expert.getRobotModel(),
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

        self._hardResetJointState(self.nao)
        self._hardResetJointState(self.nao_expert)

        pybullet.removeConstraint(
            self.balance_constraint_nao,
            physicsClientId=self.client)
        pybullet.removeConstraint(
            self.balance_constraint_nao_expert,
            physicsClientId=self.client)

        # time.sleep(0.5)

    def _termination(self):
        """
        Terminates the environment
        """
        self.simulation_manager.stopSimulation(self.client)
