#!/usr/bin/env python

import sys
sys.path.append("/usr/local/cython/")

import math

import numpy as np

import gtsam

import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot

graph = gtsam.NonlinearFactorGraph()

cov_mat = np.zeros((3,3))
np.fill_diagonal(cov_mat, [0.2*0.2,0.5*0.5,0.1*0.1])
odom_noise_model = gtsam.noiseModel_Gaussian.Covariance(cov_mat)

prior_noise_model = gtsam.noiseModel_Diagonal.Sigmas(np.array([1.6, 0.6, 0.1],dtype = np.float))
graph.add(gtsam.PriorFactorPose2(gtsam.symbol(ord('x'), 0), gtsam.Pose2(0.0, 0.0, 1.0), prior_noise_model))

graph.add(gtsam.BetweenFactorPose2(gtsam.symbol(ord('x'), 0), gtsam.symbol(ord('x'), 1), gtsam.Pose2(2.0, 0.0, 1.2), odom_noise_model))
graph.add(gtsam.BetweenFactorPose2(gtsam.symbol(ord('x'), 1), gtsam.symbol(ord('x'), 2), gtsam.Pose2(-2.0, 2.0, 1.7), odom_noise_model))

initial_estimate = gtsam.Values()
initial_estimate.insert(gtsam.symbol(ord('x'), 0), gtsam.Pose2(0.5, 0.0, 0.2))
initial_estimate.insert(gtsam.symbol(ord('x'), 1), gtsam.Pose2(2.3, 0.1, -0.2))
initial_estimate.insert(gtsam.symbol(ord('x'), 2), gtsam.Pose2(4.1, 0.1, math.pi / 2))

## LM
# params = gtsam.LevenbergMarquardtParams()
# optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
# result = optimizer.optimize()

## iSAM2
params_isam2 = gtsam.ISAM2Params()
optimizer = gtsam.ISAM2(params_isam2)
optimizer.update(graph, initial_estimate)
result = optimizer.calculateEstimate()

print("optimizer solution: \n{}".format(result))

marginals = gtsam.Marginals(graph, result)
# marginals = gtsam.Marginals(optimizer.getFactorsUnsafe(), result)
key_vec = gtsam.gtsam.KeyVector()

key_vec.push_back(gtsam.symbol(ord('x'), 0))
key_vec.push_back(gtsam.symbol(ord('x'), 1))
key_vec.push_back(gtsam.symbol(ord('x'), 2))

joint_cov_full = marginals.jointMarginalCovariance(key_vec).fullMatrix()
print("joint marginal full: {0}".format(joint_cov_full))

# joint_cov02 = marginals.jointMarginalCovariance(key_vec).at(gtsam.symbol(ord('x'), 0),gtsam.symbol(ord('x'), 2))
# eigvals,_ = np.linalg.eig(joint_cov02)

marginals = gtsam.Marginals(graph, result)
for i in range(0, 3):
    print("x{0} covariance:\n{1}\n".format(i, marginals.marginalCovariance(gtsam.symbol(ord('x'), i))))

fig = plt.figure(0)
for i in range(0, 3):
    gtsam_plot.plot_pose2(0, result.atPose2(gtsam.symbol(ord('x'), i)), 0.5, marginals.marginalCovariance(gtsam.symbol(ord('x'), i)))

plt.axis('equal')
plt.show()