import unittest
import logopy.utils.tf_utils as tf_utils

import math
import numpy as np

import torch


class TestTransforms2D(unittest.TestCase):

    def test_between_trans(self):

        pose1 = torch.FloatTensor([1, 0, 0])[None, :]
        pose2 = torch.FloatTensor([0, -1, 0])[None, :]

        pose12 = tf_utils.tf2d_between(pose1, pose2)
        pose21 = tf_utils.tf2d_between(pose2, pose1)

        pose12exp = torch.FloatTensor([-1, -1, 0])[None, :]
        pose21exp = torch.FloatTensor([1, 1, 0])[None, :]

        self.assertEqual(torch.allclose(pose12, pose12exp, 1e-5), True)
        self.assertEqual(torch.allclose(pose21, pose21exp, 1e-5), True)

    def test_between_trans_rot(self):

        pose1 = torch.FloatTensor([3, -2, math.pi])[None, :]
        pose2 = torch.FloatTensor([1, 2, math.pi/2])[None, :]

        pose12 = tf_utils.tf2d_between(pose1, pose2)
        pose21 = tf_utils.tf2d_between(pose2, pose1)

        pose12exp = torch.FloatTensor([2, -4, -1.5708])[None, :]
        pose21exp = torch.FloatTensor([-4, -2, 1.5708])[None, :]

        self.assertEqual(torch.allclose(pose12, pose12exp, 1e-5), True)
        self.assertEqual(torch.allclose(pose21, pose21exp, 1e-5), True)

    def test_roundtrip(self):

        pose1 = torch.rand(5, 3)
        pose2 = torch.rand(5, 3)

        pose12 = tf_utils.tf2d_between(pose1, pose2)
        pose2hat = tf_utils.tf2d_compose(pose1, pose12)

        self.assertEqual(torch.allclose(pose2, pose2hat, 1e-5), True)


if __name__ == '__main__':
    unittest.main()
