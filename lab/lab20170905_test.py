import unittest
import numpy as np

from lab20170905 import TrainingSet

class Test20170905(unittest.TestCase):
    def assertApproxZero(self, subject):
        self.assertTrue(-0.00001 <= subject <= 0.00001)

    def test_trivial_whiteboard_sample(self):
        set = TrainingSet(np.array([0, 1]), np.array([1, -1]))
        minimd = set.randGuessMimizes()

        self.assertTrue(minimd.success)

        cost = set.costof(minimd.x[0], minimd.x[1])
        self.assertApproxZero(cost)

    def test_trivial_always_a_line(self):
        for i in range(0, 10):
            with self.subTest(i):
                randSet = TrainingSet.buildRandomTrainer()
                minimd = randSet.randGuessMimizes()

                self.assertTrue(minimd.success)

                cost = randSet.costof(minimd.x[0], minimd.x[1])
                self.assertApproxZero(cost)

if __name__ == '__main__':
    unittest.main()
