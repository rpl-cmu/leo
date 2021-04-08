#!/usr/bin/env python

import sys
sys.path.append("/usr/local/cython/")

import math
import numpy as np
from scipy.optimize import minimize

import os
import json
import hydra
from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

import torch
from torch.autograd import Variable
import gtsam

from logopy.utils import tf_utils, dir_utils
from logopy.utils.Logger import Logger

def sampler_gaussian(mean, cov, n_samples=2):

    if cov is None:
        return mean

    cov = 1 * cov 

    n_x, dim_x = mean.shape
    xopt = np.random.multivariate_normal(np.reshape(mean, (-1)), cov, n_samples)
    xopt = np.reshape(xopt, (-1, n_x, dim_x))

    return xopt