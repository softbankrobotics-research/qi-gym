import os
import sys
import torch

try:
    import wandb
    WANDB_IMPORTED = True

except ImportError:
    WANDB_IMPORTED = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TBOARD_IMPORTED = True

except ImportError:
    TBOARD_IMPORTED = False


class LogHandler:
    """
    Wrapper class for the logs, allowing to log data through tensorboard and
    wandb
    """

    def __init__(
            self,
            project_name,
            enable_tensorboard=True,
            enable_wandb=True,
            tensorboard_folder='logs'):
        """
        Constructor

        Parameters:
            project_name - The name of the project
            enable_tensorboard - Tensorboard enabled if True, disabled if False
            enable_wandb - Wandb enabled if True, disabled if False
            tensorboard_folder - Path to the tensorboard log folder
        """
        self.tboard_writer = None
        self.wandb_writer = None
        self._wandb_enabled = False
        self._tboard_enabled = False
        self.project_name = project_name
        self.tensorboard_folder = tensorboard_folder

        self.enable_loggers(
            enable_tensorboard=enable_tensorboard,
            enable_wandb=enable_wandb)

    def enable_loggers(self, enable_tensorboard=True, enable_wandb=True):
        """
        Enables/disables the tensorboard logger and wandb logger. Both will be
        enabled by default

        Parameters:
            enable_tensorboard - Tensorboard enabled if True, disabled if False
            enable_wandb - Wandb enabled if True, disabled if False
        """
        self._tboard_enabled = enable_tensorboard
        self._wandb_enabled = enable_wandb
        self._check_imports()

    def set_experiment(self, name, description="None", config={}):
        """
        Set an experiment for the current writer

        Parameters:
            name - The name of the experiment (tensorboad board name, wandb run
            name)
            description - Additional description for the experiment (only used
            by wandb), optional
            config - The parameters passed to the program for this experiment (
            as a parsed Argparse object), optional
        """
        if self._tboard_enabled:
            self.tboard_writer = SummaryWriter(
                os.path.join(self.tensorboard_folder, name))

        if self._wandb_enabled:
            self.wandb_writer = wandb.init(
                project=self.project_name,
                name=name,
                notes=description,
                config=config)

    def log_scalar(self, name, scalar, n_iter=None, commit=True):
        """
        Log a scalar

        Parameters:
            name - Name of the plot where the scalar will be displayed
            scalar - The scalar
            n_iter - Iteration number, required by tensorboard (won't log if
            not specified)
            commit - Boolean, won't increase wandb global step if False. True
            by default
        """
        if self._wandb_enabled:
            self.wandb_writer.log({name: scalar}, commit=commit)

        if self._tboard_enabled and n_iter is not None:
            self.tboard_writer.add_scalar(name, scalar, n_iter)

    def log_scalars(self, scalar_dict, n_iter=None, commit=True):
        """
        Log scalars

        Parameters:
            scalar_dict - Dictionnary containing scalar names as keys and their
            respective values
            n_iter - Iteration number, required by tensorboard (won't log if
            not specified)
            commit - Boolean, won't increase wandb global step if False. True
            by default
        """
        if self._wandb_enabled:
            self.wandb_writer.log(scalar_dict, commit=commit)

        if self._tboard_enabled and n_iter is not None:
            for key, value in scalar_dict.items():
                self.tboard_writer.add_scalar(key, value, n_iter)

    def log_image(self, name, image, caption="", n_iter=None, commit=False):
        """
        Log an image

        Parameters:
            name - Name of the section where the image will be displayed
            image - The image (numpy array)
            n_iter - Iteration number, will be required by tensorboard (not
            currently implemented)
            commit - Boolean, will increase wandb global step if True. False
            by default
        """
        if self._wandb_enabled:
            self.wandb_writer.log(
                {name: [wandb.Image(image, caption=caption)]},
                commit=commit)

    def close(self):
        """
        Stops the wandb run if wandb is enabled
        """
        if self._wandb_enabled and self.wandb_writer is not None:
            self.wandb_writer.finish()

    def _check_imports(self):
        """
        Check that the required loggers can be imported
        """
        if not TBOARD_IMPORTED:
            self._tboard_enabled = False
            print("Can't import tensorboard")

        if not WANDB_IMPORTED:
            self._wandb_enabled = False
            print("Can't import wandb")
