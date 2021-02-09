import unittest


class DummyTest(unittest.TestCase):
    """
    Dummy unit test
    """
    @classmethod
    def setUpClass(cls):
        """
        Set up the test env
        """
        pass

    @classmethod
    def tearDownClass(cls):
        """
        Tear down the test env
        """
        pass

    def test_dummy(self):
        """
        Dummy test
        """
        self.assertTrue(True)
