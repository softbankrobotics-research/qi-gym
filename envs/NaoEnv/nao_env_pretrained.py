#!/usr/bin/env python
# coding: utf-8
import sys
CV2_ROS = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if CV2_ROS in sys.path:
    sys.path.remove(CV2_ROS)
    sys.path.append(CV2_ROS)
import time
from nao_env import NaoEnv
import pybullet
import json


class NaoEnvPretrained(NaoEnv):
    """
    Gym environment for the NAO robot, walking task
    """
    def __init__(self, gui=False):
        NaoEnv.__init__(self, gui)
        self.data = None
        with open("data/nao/trajectories_walk.json") as json_file:
            self.data = json.load(json_file)

        self.positions = self.data["trajectories"].copy()
        # self.positions_copy = self.positions[0].copy()
        # self.positions.pop(0)
        self.action_list = list()
        self.last_time_action = 0

    def reset(self):
        """
        Resets the environment for a new episode
        """
        obs = NaoEnv.reset(self)
        self.positions_copy = self.positions[0].copy()
        self.actual_traj = self.positions[0].copy()
        self.positions.pop(0)
        self.last_time_action = 0
        return obs

    def _setVelocities(self, joints, n_velocities):
        """
        Sets velocities on the robot joints
        """
        if len(self.positions_copy) == 0:
            self.episode_over = True
            return
        for joint, position in zip(
                joints, self.positions_copy[0]):
            self.nao.setAngles(joint, position, 1.0)
        self.positions_copy.pop(0)

    def _setPositions(self, joints, n_velocities):
        """
        Sets velocities on the robot joints
        """
        if len(self.positions_copy) == 0:
            self.episode_over = True
            return
        for joint, position in zip(
                self.controlled_joints, self.positions_copy[0]):
            pybullet.setJointMotorControl2(
                self.nao.robot_model,
                self.nao.joint_dict[joint].getIndex(),
                pybullet.POSITION_CONTROL,
                targetPosition=position,
                force=self.nao.joint_dict[joint].getMaxEffort(),
                physicsClientId=self.client)
        self.positions_copy.pop(0)

    def walking_expert_speed(self, _obs):
        """
        Generates actions accordingly to the obs
        """
        actions = list()
        for name in self.controlled_joints:
            _, vel, _ = self._getJointState(name)
            actions.append(vel)
        self.action_list.append(actions)
        return actions

    def walking_expert_position(self, _obs):
        """
        Generates actions accordingly to the obs
        """
        actions = list()
        for name in self.controlled_joints_reduced:
            upper = self.nao.joint_dict[name].getUpperLimit()
            lower = self.nao.joint_dict[name].getLowerLimit()
            position = self.nao.getAnglesPosition(name)
            numerator = (position - lower)
            denominator = (upper - lower)
            actions.append(2 * (numerator / denominator) - 1)
        self.action_list.append(actions)
        return actions
