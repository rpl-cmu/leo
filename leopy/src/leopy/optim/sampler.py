import numpy as np
from leopy.utils import tf_utils

def compute_x_samples(mean, dx_samples):

    num_samples, num_x, dim_x = dx_samples.shape
    x_samples = np.zeros(dx_samples.shape)

    for sidx in range(0, num_samples):
        for xidx in range(0, num_x):
            x_mean = tf_utils.vec3_to_pose2(mean[xidx, :])
            x_sample_pose = x_mean.retract(dx_samples[sidx, xidx, :])
            x_samples[sidx, xidx, :] = tf_utils.pose2_to_vec3(x_sample_pose)

    return x_samples

def sampler_gaussian(mean, cov, n_samples=1, temp=1.):
    
    if cov is None:
        return mean

    num_x, dim_x = mean.shape
    mean_vec = np.reshape(mean, -1)

    cov = cov / temp

    dx_samples = np.random.multivariate_normal(np.zeros(mean_vec.shape), cov, n_samples)
    dx_samples = dx_samples.reshape(dx_samples.shape[0], num_x, dim_x)

    # x_samples: num_samples, num_x, dim_x
    x_samples = compute_x_samples(mean, dx_samples)

    return x_samples