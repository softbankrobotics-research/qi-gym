import gym


class QiEnv(gym.Env):
    """
    Gym environment with qiBullet functionalities
    """

    def __init__(self, gui=False):
        """
        Constructor
        """
        super(QiEnv, self).__init__()
        self.action_space = None
        self.observation_space = None

    def step(self, action):
        """
        ABSTRACT method, needs to be redefined by the child environment. Steps
        the environment

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
        return NotImplementedError

    def reset(self):
        """
        ABSTRACT method, needs to be redefined by the child environment. Resets
        the environment for a new episode

        Returns:
            observation - An environment-specific object representing the
            observation of the environment. (object)
        """
        return NotImplementedError

    def render(self, mode='human'):
        """
        ABSTRACT method, needs to be redefined by the child environment.
        Performs a render operation in the environment
        """
        return NotImplementedError

    def close(self):
        """
        ABSTRACT method, needs to be redefined by the child environment.
        Terminates the environment
        """
        return NotImplementedError
