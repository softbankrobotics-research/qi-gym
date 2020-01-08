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

OBS_DIM = 59


class NaoEnvPretrained(NaoEnv):
    """
    Gym environment for the NAO robot, walking task
    """
    def __init__(self, gui=False):
        NaoEnv.__init__(self, gui)
        infile = open("data/nao/walk_positions.pckl", 'rb')
        self.positions = pickle.load(infile)
        self.positions_copy = list()
        infile.close()
        self.action_list = list()
        self.last_time_action = time.time()

    def reset(self):
        """
        Resets the environment for a new episode
        """
        self.episode_over = False
        self.previous_x = 0
        self.counter = 0
        self.number_of_step_in_episode = 0
        self.last_time_action = time.time()
        self.foot_step_number = 0
        self.feet_ahead = None
        self._resetScene()

        self.positions_copy = self.positions.copy()

        obs, _ = self._getState()
        return obs

    def _setVelocities(self, joints, n_velocities):
        """
        Sets velocities on the robot joints
        """
        if len(self.positions_copy) == 0:
            self.episode_over = True
            return
        if time.time() - self.last_time_action >= 0.0015:
            self.last_time_action = time.time()
            for joint, position in zip(
                    self.controlled_joints, self.positions_copy[0]):
                self.nao.setAngles(joint,
                                   position, 1.0)
            self.positions_copy.pop(0)

    def walking_expert_speed(self, _obs):
        """
        Generates actions accordingly to the obs
        """
        actions = list()

        for name in self.controlled_joints:
            actions.append(
                self.nao.getAnglesVelocity(name) /
                self.nao.joint_dict[name].getMaxVelocity())
        self.action_list.append(actions)
        return actions

    def walking_expert_position(self, _obs):
        """
        Generates actions accordingly to the obs
        """
        actions = list()

        for name in self.controlled_joints:
            actions.append(
                self.nao.getAnglesPosition(name))
        return actions