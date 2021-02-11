import os
import shutil
import torch


class FileManager:
    """
    Filesystem management for the project. Will allow create missing
    dictionaries, create folders related to a specific run, etc.
    """

    def __init__(
            self,
            checkpoints_folder='checkpoints',
            tensorboard_folder='logs'):
        """
        Constructor

        Parameters:
            checkpoints_folder - Name folder containing the checkpoints
            tensorboard_folder - Name of the folder containing the tensorboard
            logs
        """
        self._experiment_name = None
        self.checkpoints = checkpoints_folder
        self.tensorboard = tensorboard_folder

    def get_experiment_name(self):
        """
        Getter for the name of the experiment

        Returns:
            experiment_name - The name of the current experiment, None by
            default
        """
        return self._experiment_name

    def filesystem_check(self):
        """
        Will check that the checkpoints and logs folders exist. If not, the
        folders will be created
        """
        if not os.path.isdir(self.checkpoints):
            os.mkdir(self.checkpoints)
            print("Checkpoints folder created: " + self.checkpoints)

        if not os.path.isdir(self.tensorboard):
            os.mkdir(self.tensorboard)
            print("Tensorboard logs folder created: " + self.tensorboard)

    def create_new_experiment(self, experiment_name=None):
        """
        Creates a new experiment. The name of the experiment is
        experiment_name. By default, the experiment will be named experiment_x,
        x being the number of the current experiment (if another experiment is
        found with the number a, x = a + 1). The method will create a dedicated
        folder in the checkpoints folder and return the chosen experiment_name

        Parameters:
            experiment_name - A specific exprience_name can be specified by the
            user

        Returns:
            experiment_name - The name of the created experiment (useful if the
            user didn't give a specific experiment name)
        """
        if experiment_name is not None:
            # Check that the experiment does not exist
            if self._check_experiment_exists(experiment_name):
                return None

            else:
                self._experiment_name = experiment_name
                os.mkdir(os.path.join(self.checkpoints, experiment_name))
                return self._experiment_name

        # If the name of the experiment isn't specified
        experiment_numbers = [int(x.split('_')[1]) for x in
                              self._get_default_experiment_folders()]

        if len(experiment_numbers) == 0:
            self._experiment_name = 'experiment_0'
        else:
            self._experiment_name = 'experiment_' + str(
                max(experiment_numbers) + 1)

        os.mkdir(os.path.join(self.checkpoints, self._experiment_name))
        return self._experiment_name

    def clear_experiment(self, experiment_name):
        """
        Removes an experiment from the checkpoints folder.

        Parameters:
            experiment_name - The name of the experiment

        Returns:
            success - Boolean, True if the clear was successful, False
            otherwise
        """
        if self._check_experiment_exists(experiment_name):
            shutil.rmtree(os.path.join(self.checkpoints, experiment_name))
            return True
        else:
            return False

    def clear_all_experiments(self):
        """
        Clear all the experiments from the checkpoints folder (the ones with
        default names and the ones with personalized names)
        """
        for experiment in self._get_all_experiment_folders():
            shutil.rmtree(os.path.join(self.checkpoints, experiment))

    def clear_default_experiments(self):
        """
        Clear the default experiments from the checkpoints folder (the ones
        with default names)
        """
        for experiment in self._get_default_experiment_folders():
            shutil.rmtree(os.path.join(self.checkpoints, experiment))

    def save_checkpoint(self, name, model, optimizer=None):
        """
        Save a checkpoint for the current experiment. If no current experimen
        is defined, the method will return False. If saving the checkpoint
        succeeds, the method will return True

        Parameters:
            name - The name of the chekpoint
            model - The model to be saved (its state dict will be saved)
            optimizer - The optimizer (its state dict will be saved), optional

        Returns:
            success - Boolean, True if the save was successful, False
            otherwise
        """
        if self._experiment_name is None:
            return False

        checkpoint_dict = {'model': model.state_dict()}

        if optimizer is not None:
            checkpoint_dict['optimizer': optimizer.state_dict()]

        torch.save(
            checkpoint_dict,
            os.path.join(self.checkpoints, self._experiment_name, name))

        return True

    def load_checkpoint(self, name, experiment):
        """
        Loads a checkpoint. We assume that the checkpoint is located in the
        checkpoint folder. The model and optimizer state dicts are returned as
        a tuple (model_state_dict, optimizer_state_dict). Be aware that the
        state dict of the optimizer might not be specified, in which case the
        second element of the tuple will be set to None. If the specified
        experiment or the specified checkpoint name do not exist, the method
        will raise a FileNotFoundError

        Parameters:
            name - The name of the checkpoint
            experiment - The name of the experiment

        Returns:
            model - The state dict of the checkpoint's model
            optimizer - The state dict of the checkpoint's model if provided,
            None otherwise
        """
        checkpoint = torch.load(
            os.path.join(self.checkpoints, experiment, name))

        if 'optimizer' in checkpoint:
            return (checkpoint['model'], checkpoint['optimizer'])
        else:
            return (checkpoint['model'], None)

    def _get_all_experiment_folders(self):
        """
        Returns a list containing the name of all the folders in
        checkpoints (all of the folder are considered as experiments)

        Returns:
            folders - List containing the name of the all the experiment
            folders
        """
        return [x for x in os.listdir(self.checkpoints)
                if self._check_experiment_exists(x)]

    def _get_default_experiment_folders(self):
        """
        Returns a list containing the name of the default experiment folders in
        checkpoints (the default folder name for an experiment is experiment_x)

        Returns:
            folders - List containing the name of the default experiment
            folders
        """
        return [x for x in self._get_all_experiment_folders() if
                x.split('_')[0] == 'experiment']

    def _check_experiment_exists(self, experiment):
        """
        Checks that an experiment exists in the checkpoints folder.

        Parameters:
            experiment - The name of the experiment

        Returns:
            exists - Boolean, True if the experiment exists, False otherwise
        """
        return os.path.isdir(os.path.join(self.checkpoints, experiment))
