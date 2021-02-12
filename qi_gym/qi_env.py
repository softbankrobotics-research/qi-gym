import gym
import time
import pybullet
from qibullet import SimulationManager


class QiEnv(gym.Env):
    """
    Gym environment with qiBullet functionalities, providing helpful functions
    """

    def __init__(self):
        """
        Constructor
        """
        super(QiEnv, self).__init__()
        self.simulation_manager = SimulationManager()

        # The action space and the observation have to be specified in the
        # child class
        self.action_space = None
        self.observation_space = None

        self.episode_start_time = None
        self.episode_over = False
        self.episode_failed = False
        self.episode_reward = 0.0
        self.episode_number = 0
        self.episode_steps = 0

        self.spawn_scene()

    def step(self, action):
        """
        Steps the environment

        Parameters:
            action - The action to be performed

        Returns:
            observation - An environment-specific object representing the
            observation of the environment. (object)
            reward - Amount of reward achieved by the previous action. The
            scale varies between environments, but the goal is always to
            increase your total reward (float)
            done - Whether it's time to reset the environment again. Most (but
            not all) tasks are divided up into well-defined episodes, and done
            being True indicates the episode has terminated (for example,
            perhaps the pole tipped too far, or you lost your last life) (bool)
            info - Diagnostic information useful for debugging. It can
            sometimes be useful for learning (for example, it might contain the
            raw probabilities behind the environment's last state change)
            However, official evaluations of your agent are not allowed to use
            this for learning (dict)
        """
        self.episode_steps += 1
        self.apply_action(action)
        observation, reward = self.compute_state()

        return observation, reward, self.is_episode_over(), {}

    def reset(self):
        """
        Resets the environment for a new episode

        Returns:
            observation - An environment-specific object representing the
            observation of the environment. (object)
        """
        self._reset_metrics()
        self.reset_scene()

        # Reset the start time for the current episode
        self.episode_start_time = time.time()

        return self.get_observation()

    def render(self, mode='human'):
        """
        Performs a render operation in the environment
        """
        pass

    def close(self, client):
        """
        Terminates the environment
        """
        self.simulation_manager.stopSimulation(client)

    def apply_action(self, action):
        """
        ABSTRACT method, needs to be redefined by the child environment. Apply
        an action to the environment
        """
        return NotImplementedError

    def compute_state(self):
        """
        ABSTRACT method, needs to be redefined by the child environment.
        This method will compute and return the current reward of the
        envionment, along with an observation of the enrironment. The method
        will also determine wether if the episode is over or not.
        """
        return NotImplementedError

    def get_observation(self):
        """
        ABSTRACT method, needs to be redefined by the child environment.
        Returns the observation for the environment

        Returns:
            observation - The observation
        """
        return NotImplementedError

    def spawn_scene(self):
        """
        ABSTRACT method, needs to be redefined by the child environment.
        Use this method to spawn a virtual scene. This method will be used
        once, when creating the environment
        """
        return NotImplementedError

    def reset_scene(self):
        """
        ABSTRACT method, needs to be redefined by the child environment.
        This method will be used when resetting the environment, its goal it to
        reset the state of the elements in the virtual scene
        """
        return NotImplementedError

    def set_episode_over(self, episode_over):
        """
        Specifies wether if the episode is over or not

        Parameters:
            episode_over - Boolean, if True specifies that the episode is over
        """
        self.episode_over = episode_over

    def is_episode_over(self):
        """
        Will return True if the episode is over, false otherwise

        Returns:
            episode_over - Boolean, True if the episode is over, False
            otherwise
        """
        return self.episode_over

    def hard_reset_joints(self, robot, joint_dict):
        """
        Performs a hard reset on the joints of the robot, avoiding the robot to
        get stuck in a position. To be called when setting up the scene, or
        resetting a scene

        Parameters:
            robot - Virtual qiBullet robot
            joint_dict - Dict containing the initial joint parameters,
            formatted as {'joint_name': joint_position, ...}
        """
        for name, position in joint_dict.items():
            pybullet.setJointMotorControl2(
                robot.getRobotModel(),
                robot.getJoint(name).getIndex(),
                pybullet.VELOCITY_CONTROL,
                targetVelocity=0.0)

            pybullet.resetJointState(
                robot.getRobotModel(),
                robot.getJoint(name).getIndex()
                position)

    def get_object_position(self, bullet_object):
        """
        Returns the position of the base of a Bullet object in the world frame

        Parameters:
            object - a Bullet object

        Returns:
            translation - A vector of 3 elements containing the translation
            coordinates of the object in the world frame
            quaternion - A vector of 4 elements containing the rotation of the
            object in the world frame, formatted as a quaternion
        """
        translation, quaternion = pybullet.getBasePositionAndOrientation(
            bullet_object)

        return translation, quaternion

    def _reset_metrics(self):
        """
        INTERNAL METHOD, resets variables such as episode_over. To be called in
        the reset method
        """
        self.set_episode_over(False)
        self.episode_failed = False
        self.episode_number = 0.0
        self.episode_steps = 0
