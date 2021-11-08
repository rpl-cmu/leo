import sys
sys.path.append("/usr/local/cython/")

import numpy as np
import time
import pandas as pd

from leopy.optim.nav2d_factors import *
# from leopy.optim.push2d_factors import *

from leopy.utils import tf_utils, dir_utils, vis_utils
from leopy.utils.logger import Logger

import logging
log = logging.getLogger(__name__)
class GraphOpt():
    def __init__(self):

        self.graph = gtsam.NonlinearFactorGraph()
        self.init_vals = gtsam.Values()

        self.optimizer = self.init_isam2()
        self.est_vals = gtsam.Values()

        self.logger = Logger()
        self.dataframe = pd.DataFrame()
        # self.graph_full = gtsam.NonlinearFactorGraph()

    def init_isam2(self):
        params_isam2 = gtsam.ISAM2Params()
        params_isam2.setRelinearizeThreshold(0.01)
        params_isam2.setRelinearizeSkip(10)

        return gtsam.ISAM2(params_isam2)

def init_vars_step(graphopt, tstep, dataset_type):

    if (dataset_type == "push2d"):
        key_tm1 = gtsam.symbol(ord('o'), tstep-1)
        key_t = gtsam.symbol(ord('o'), tstep)
        graphopt.init_vals.insert(key_t, graphopt.est_vals.atPose2(key_tm1))

        key_tm1 = gtsam.symbol(ord('e'), tstep-1)
        key_t = gtsam.symbol(ord('e'), tstep)
        graphopt.init_vals.insert(key_t, graphopt.est_vals.atPose2(key_tm1))
    else:
        key_tm1 = gtsam.symbol(ord('x'), tstep-1)
        key_t = gtsam.symbol(ord('x'), tstep)
        graphopt.init_vals.insert(key_t, graphopt.est_vals.atPose2(key_tm1))

    return graphopt


def reset_graph(graphopt):
    # graphopt.graph_full.push_back(graphopt.graph)
    
    graphopt.graph.resize(0)
    graphopt.init_vals.clear()

    return graphopt


def optimizer_update(graphopt):
    graphopt.optimizer.update(graphopt.graph, graphopt.init_vals)
    graphopt.est_vals = graphopt.optimizer.calculateEstimate()

    return graphopt

def print_step(tstep, data, isam2_estimate, dataset_type):

    if (dataset_type == "push2d"):
        key = gtsam.symbol(ord('o'), tstep)
        print('Estimated, Grountruth pose: \n {0} \n {1} '.format(
            tf_utils.pose2_to_vec3(isam2_estimate.atPose2(key)), data['obj_poses_gt'][tstep]))
    else:
        key = gtsam.symbol(ord('x'), tstep)
        print('Estimated, Grountruth pose: \n {0} \n {1} '.format(
            tf_utils.pose2_to_vec3(isam2_estimate.atPose2(key)), data['poses_gt'][tstep]))

def log_step(tstep, logger, data, optimizer, dataset_type):

    if (dataset_type == 'nav2d'):
        logger = log_step_nav2d(tstep, logger, data, optimizer)
    elif (dataset_type == 'push2d'):
        logger = log_step_push2d(tstep, logger, data, optimizer)
    
    return logger

def log_step_df(tstep, dataframe, data, optimizer, dataset_type):

    if (dataset_type == 'nav2d'):
        logger = log_step_df_nav2d(tstep, dataframe, data, optimizer)
    elif (dataset_type == 'push2d'):
        logger = log_step_df_push2d(tstep, dataframe, data, optimizer)
    
    return logger

def add_first_pose_priors(graphopt, data, dataset_type):

    prior_cov = np.array([1e-6, 1e-6, 1e-6])

    if(dataset_type == 'nav2d'):
        keys = [gtsam.symbol(ord('x'), 0)]
        graphopt.init_vals.insert(
            keys[0], tf_utils.vec3_to_pose2(data['poses_gt'][0]))
        graphopt.graph = add_unary_factor(
            graphopt.graph, keys, prior_cov, data['poses_gt'][0])
    elif (dataset_type == 'push2d'):
        keys = [gtsam.symbol(ord('o'), 0)]
        graphopt.init_vals.insert(
            keys[0], tf_utils.vec3_to_pose2(data['obj_poses_gt'][0]))
        graphopt.graph = add_unary_factor(
            graphopt.graph, keys, prior_cov, data['obj_poses_gt'][0])
        keys = [gtsam.symbol(ord('e'), 0)]
        graphopt.init_vals.insert(
            keys[0], tf_utils.vec3_to_pose2(data['ee_poses_gt'][0]))
        graphopt.graph = add_unary_factor(
            graphopt.graph, keys, prior_cov, data['ee_poses_gt'][0])

    graphopt = optimizer_update(graphopt)
    graphopt = reset_graph(graphopt)

    return graphopt


def get_optimizer_soln(tstep, graphopt, dataset_type, sampler=False):

    poses_graph = graphopt.optimizer.calculateEstimate()
    pose_vec_graph = np.zeros((poses_graph.size(), 3))    

    num_poses = tstep + 1
    if (dataset_type == 'nav2d'):
        keys = [[gtsam.symbol(ord('x'), i)] for i in range(0, num_poses)]
    elif (dataset_type == 'push2d'):
        keys = [[gtsam.symbol(ord('o'), i), gtsam.symbol(ord('e'), i)] for i in range(0, num_poses)] 

    key_vec = gtsam.gtsam.KeyVector()
    for key_idx in range(0, len(keys[0])):
        for pose_idx in range(0, num_poses):
            key = keys[pose_idx][key_idx]
            pose2d = poses_graph.atPose2(key)
            pose_vec_graph[key_idx * num_poses + pose_idx, :] = [pose2d.x(), pose2d.y(), pose2d.theta()]

            key_vec.push_back(key)

    mean = pose_vec_graph
    cov = None
    
    if sampler:
        marginals = gtsam.Marginals(graphopt.optimizer.getFactorsUnsafe(), poses_graph)
        cov = marginals.jointMarginalCovariance(key_vec).fullMatrix()
    
    return (mean, cov)

def run_optimizer(cost, params=None):
    ''' construct graph from data and optimize '''

    dataset_type = params.dataio.dataset_type

    graphopt = GraphOpt()
    graphopt = add_first_pose_priors(graphopt, cost.data, dataset_type)

    if (dataset_type == "push2d"):
        nsteps = np.minimum(params.optim.nsteps, len(cost.data['obj_poses_gt']))

        planar_sdf = PlanarSDF("{0}/{1}/sdf/{2}.json".format(
            params.BASE_PATH, params.dataio.srcdir_dataset, params.dataio.obj_shape))
        tactile_model = torch.jit.load("{0}/local/models/{1}.pt".format(
            params.BASE_PATH, params.tactile_model.name))
    else:
        nsteps = np.minimum(params.optim.nsteps, len(cost.data['poses_gt']))

    for tstep in range(1, nsteps):

        # filter out curr step factors
        factor_idxs = [idxs for idxs, keys in enumerate(
            cost.data['factor_keyids']) if (max(keys) == tstep)]

        factor_keysyms = [cost.data['factor_keysyms'][idx] for idx in factor_idxs]
        factor_keyids = [cost.data['factor_keyids'][idx] for idx in factor_idxs]
        factor_meas = [cost.data['factor_meas'][idx] for idx in factor_idxs]
        factor_names = [cost.data['factor_names'][idx] for idx in factor_idxs]

        # compute factor costs
        for idx in range(0, len(factor_idxs)):

            key_syms, key_ids = factor_keysyms[idx], factor_keyids[idx]

            keys = [gtsam.symbol(ord(key_syms[i]), key_id)
                    for i, key_id in enumerate(key_ids)]
            factor_name = factor_names[idx]

            sigma_inv_val = cost.theta.get_sigma_inv(factor_name, factor_meas[idx])
            sigma_inv_val = sigma_inv_val.detach().cpu().numpy()
            # factor_cov = np.reciprocal(sigma_inv_val) # factor_cov: sigma format
            factor_cov = np.reciprocal(np.sqrt(sigma_inv_val)) # factor_cov: sigma_sq format

            if (factor_name == 'gps'):
                graphopt.graph = add_unary_factor(
                    graphopt.graph, keys, factor_cov, factor_meas[idx][0:3])
            elif (factor_name == 'odom'):
                graphopt.graph = add_binary_odom_factor(
                    graphopt.graph, keys, factor_cov, factor_meas[idx][0:3])
            elif (factor_name == 'ee_pose_prior'):
                graphopt.graph = add_unary_factor(
                    graphopt.graph, keys, factor_cov, factor_meas[idx])
            elif (factor_name == 'qs_push_motion'):
                graphopt.graph = add_qs_motion_factor(
                    graphopt.graph, keys, factor_cov, params=params)
            elif (factor_name == 'sdf_intersection'):
                graphopt.graph = add_sdf_intersection_factor(
                    graphopt.graph, keys, factor_cov, planar_sdf.sdf)
            elif (factor_name == 'tactile_rel'):
                if (params.tactile_model.oracle == True):
                    tf_pred_net = tactile_oracle_output(cost.data, key_ids)
                else:
                    if (params.dataio.model_type == "fixed_cov_varying_mean"):
                        tf_pred_net = tactile_model_output(
                            cost, img_feats=factor_meas[idx], params=params.tactile_model) # learnable model weights
                        tf_pred_net[3] = np.clip(tf_pred_net[3], -1., 1.)
                    else:
                        tf_pred_net = tactile_model_output_fixed(
                            tactile_model, img_feats=factor_meas[idx], params=params.tactile_model) # fixed model weights
                graphopt.graph = add_tactile_rel_meas_factor(
                    graphopt.graph, keys, factor_cov, tf_pred_net, params=params.tactile_model)
            elif (factor_name == 'binary_interseq_obj'):
                graphopt.graph = add_binary_odom_factor(
                    graphopt.graph, keys, factor_cov, factor_meas[idx])

        # init variables
        graphopt = init_vars_step(graphopt, tstep, dataset_type)

        # optimize
        graphopt = optimizer_update(graphopt)

        # print step
        # print_step(tstep, cost.data, graphopt.est_vals, dataset_type)

        # log step
        # todo: phase out log_step and only use log_step_df. Currently log_step needed for plotting.
        graphopt.logger = log_step(
            tstep, graphopt.logger, cost.data, graphopt.optimizer, dataset_type)        

        if params.logger.enable:
            graphopt.dataframe = log_step_df(
                tstep, graphopt.dataframe, cost.data, graphopt.optimizer, dataset_type)
 
        # vis step
        if (params.optim.vis_step):
            vis_utils.vis_step(tstep, graphopt.logger, params=params)

        # reset graph
        graphopt = reset_graph(graphopt)

    mean, cov = get_optimizer_soln(tstep, graphopt, dataset_type, params.leo.sampler)

    # log final solution
    if params.logger.enable:
        cov_log = cov.tolist() if cov is not None else cov
        data_row = pd.DataFrame({'opt/mean': [mean.tolist()], 'opt/covariance': [cov_log]}, index=[tstep], dtype=object)
        data_row.index.name = 'tstep'
        graphopt.dataframe = pd.concat([graphopt.dataframe, data_row], axis=1)

    if (params.optim.save_fig) | (params.optim.show_fig):
        vis_utils.vis_step(tstep, graphopt.logger, params=params)

    # if params.optim.verbose:
    #     graph_full = graphopt.optimizer.getFactorsUnsafe()
    #     graph_cost = graph_full.error(graphopt.optimizer.calculateEstimate())
    #     print("gtsam cost: {0}".format(graph_cost))
    #     # print("gtsam per factor cost for {0} factors over {1} time steps".format(
    #     # graph_full.size(), nsteps))
    #     # for i in range(graph_full.size()):
    #     #     print(graph_full.at(i).error(graphopt.optimizer.calculateEstimate()))
    
    return (mean, cov, graphopt.dataframe)
