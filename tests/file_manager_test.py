import unittest
import shutil

from qi_gym import FileManager


class FileManagerTest(unittest.TestCase):
    """
    File manager tests
    """
    @classmethod
    def setUpClass(cls):
        """
        Set up the test env
        """
        cls.file_manager = FileManager()
        cls.file_manager.filesystem_check()

    @classmethod
    def tearDownClass(cls):
        """
        Tear down the test env
        """
        cls.file_manager.clear_all_experiments()
        shutil.rmtree(cls.file_manager.checkpoints)
        shutil.rmtree(cls.file_manager.tensorboard)

    def test_experiment_creation_deletion(self):
        """
        Test the experiment creation methods, along with the experiment
        deletion methods
        """
        # Test the experiment creation with a default name and a specific name
        default_name = FileManagerTest.file_manager.create_new_experiment()
        specific_name = FileManagerTest.file_manager.create_new_experiment(
            experiment_name="specific_name")

        self.assertIsInstance(default_name, str)
        self.assertIsInstance(specific_name, str)

        # Check that the created experiment has an associated folder
        self.assertTrue(FileManagerTest.file_manager._check_experiment_exists(
            default_name))
        self.assertTrue(FileManagerTest.file_manager._check_experiment_exists(
            specific_name))

        # Check that when creating an experiment with the same name, the method
        # returns None
        self.assertIsNone(FileManagerTest.file_manager.create_new_experiment(
            experiment_name=default_name))
        self.assertIsNone(FileManagerTest.file_manager.create_new_experiment(
            experiment_name=specific_name))

        # Remove the experiments
        FileManagerTest.file_manager.clear_experiment(default_name)
        FileManagerTest.file_manager.clear_experiment(specific_name)

        # Check that the deleted experiments where correctly deleted
        self.assertFalse(FileManagerTest.file_manager._check_experiment_exists(
            default_name))
        self.assertFalse(FileManagerTest.file_manager._check_experiment_exists(
            specific_name))

    def test_get_experiment_name(self):
        """
        Test the get experiment name method
        """
        name = FileManagerTest.file_manager.create_new_experiment()
        self.assertEqual(
            name,
            FileManagerTest.file_manager.get_experiment_name())

        FileManagerTest.file_manager.clear_experiment(name)

    def test_general_experiment_clearing(self):
        """
        Test removing batch of experiments
        """
        default_names = list()
        specific_names = ['test', 'other_test', 'another_experiment']

        for i in range(len(specific_names)):
            default_names.append(
                FileManagerTest.file_manager.create_new_experiment())

            FileManagerTest.file_manager.create_new_experiment(
                specific_names[i])

            self.assertTrue(FileManagerTest.file_manager._check_experiment_exists(
                default_names[i]))
            self.assertTrue(FileManagerTest.file_manager._check_experiment_exists(
                specific_names[i]))

        # Delete default experiments
        FileManagerTest.file_manager.clear_default_experiments()

        for name in default_names:
            self.assertFalse(
                FileManagerTest.file_manager._check_experiment_exists(name))

        # Delete all experiments (recreate default ones before)
        default_names = list()

        for i in range(len(specific_names)):
            default_names.append(
                FileManagerTest.file_manager.create_new_experiment())

        FileManagerTest.file_manager.clear_all_experiments()

        for d_name, s_name in zip(default_names, specific_names):
            self.assertFalse(
                FileManagerTest.file_manager._check_experiment_exists(d_name))
            self.assertFalse(
                FileManagerTest.file_manager._check_experiment_exists(s_name))
