#!/usr/bin/env python
# coding: utf-8
import sys
CV2_ROS = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if CV2_ROS in sys.path:
    sys.path.remove(CV2_ROS)
    sys.path.append(CV2_ROS)
import time
from envs import NaoEnv
import pickle
import pybullet

mode = 0
# mode = 1



class NaoEnvPretrained(NaoEnv):
    """
    Gym environment for the NAO robot, walking task
    """
    def __init__(self, gui=False):
        NaoEnv.__init__(self, gui)
        infile = None
        if mode:
            # infile = open("data/nao/walk_positions_no_latency.pckl", 'rb')
            # infile = open("data/nao/walk_positions_10HZ.pckl", 'rb')
            infile = open("data/nao/walk_positions_300HZ.pckl", 'rb')
        else:
            # infile = open("data/nao/walk_positions.pckl", 'rb')
            infile = open("data/nao/walk_positions_10HZ.pckl", 'rb')
            # infile = open("data/nao/walk_positions_no_latency.pckl", 'rb')
            
        self.positions = pickle.load(infile)
        self.positions_copy = list()
        infile.close()
        self.action_list = list()
        self.last_time_action = 0
        self.outfile = None
        # if not mode:
        #     self.outfile = open("data/nao/walk_positions_300HZ.pckl", 'wb')

    def reset(self):
        """
        Resets the environment for a new episode
        """
        if self.outfile is not None and not mode and len(self.action_list) != 0:
            pickle.dump(self.action_list, self.outfile)
        obs = NaoEnv.reset(self)
        self.positions_copy = self.positions.copy()
        self.last_time_action = 0
        return obs

    def _setVelocities(self, joints, n_velocities):
        """
        Sets velocities on the robot joints
        """
        if len(self.positions_copy) == 0:
            self.episode_over = True
            return
        if not mode:
            for joint, position in zip(
                    joints, self.positions_copy[0]):
                self.nao.setAngles(joint,
                                position, 1.0)
            self.positions_copy.pop(0)
        else:
            for joint, n_pos in zip(joints, self.positions_copy[0]):
                upper = self.nao.joint_dict[joint].getUpperLimit()
                lower = self.nao.joint_dict[joint].getLowerLimit()
                pos = n_pos * (upper - lower) + lower
                pybullet.setJointMotorControl2(
                    self.nao.robot_model,
                    self.nao.joint_dict[joint].getIndex(),
                    pybullet.POSITION_CONTROL,
                    targetPosition=pos,
                    force=self.nao.joint_dict[joint].getMaxEffort(),
                    physicsClientId=self.client)
            self.positions_copy.pop(0)

    def _setPositions(self, joints, n_velocities):
        """
        Sets velocities on the robot joints
        """
        if len(self.positions_copy) == 0:
            self.episode_over = True
            return
        if not mode:
            # if time.time() - self.last_time_action >= 1/10.0:
            #     self.last_time_action = time.time()
            for joint, position in zip(
                    joints, self.positions_copy[0]):
                pybullet.setJointMotorControl2(
                    self.nao.robot_model,
                    self.nao.joint_dict[joint].getIndex(),
                    pybullet.POSITION_CONTROL,
                    targetPosition=position,
                    force=self.nao.joint_dict[joint].getMaxEffort(),
                    physicsClientId=self.client)
            self.positions_copy.pop(0)
        else:
            for joint, n_pos in zip(joints, self.positions_copy[0]):
                upper = self.nao.joint_dict[joint].getUpperLimit()
                lower = self.nao.joint_dict[joint].getLowerLimit()
                pos = n_pos * (upper - lower) + lower
                pybullet.setJointMotorControl2(
                    self.nao.robot_model,
                    self.nao.joint_dict[joint].getIndex(),
                    pybullet.POSITION_CONTROL,
                    targetPosition=pos,
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

        for name in self.controlled_joints:
            upper = self.nao.joint_dict[name].getUpperLimit()
            lower = self.nao.joint_dict[name].getLowerLimit()
            actions.append(
                (self.nao.getAnglesPosition(name) - lower)/
                (upper - lower))
        # if self.last_time_action == 0:
        #     self.action_list.append(actions)
        #     self.last_time_action = time.time()
        # if time.time() - self.last_time_action >= 1/10.0:
        #     self.last_time_action = time.time()
        self.action_list.append(actions)
        return actions