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
        self.action_list = list()

    def _setPositions(self, virtual_robot_list, joints, n_positions_list):
        """
        Sets velocities on the robot joints
        """
        index = self.index_joint_list.copy()
        joint_force = self.joint_force_list.copy()
        actions_list = [n_positions_list[1], n_positions_list[1]]
        for virtual_robot, actions in zip(virtual_robot_list, actions_list):
            pybullet.setJointMotorControlArray(
                    virtual_robot.getRobotModel(),
                    index,
                    pybullet.POSITION_CONTROL,
                    targetPositions=actions,
                    forces=joint_force,
                    physicsClientId=self.client)

    def walking_expert_position(self, _obs):
        """
        Generates actions accordingly to the obs
        """
        actions = list()
        for name in self.controlled_joints:
            upper = self.nao.joint_dict[name].getUpperLimit()
            lower = self.nao.joint_dict[name].getLowerLimit()
            position = self.nao.getAnglesPosition(name)
            numerator = (position - lower)
            denominator = (upper - lower)
            actions.append(2 * (numerator / denominator) - 1)
        self.action_list.append(actions)
        return actions
