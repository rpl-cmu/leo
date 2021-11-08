import numpy as np

import torch
import torch.nn as nn
import collections

import pytorch_lightning as pl

from leopy.utils import tf_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# theta_exp used when realizability_coeff > 0 
def add_theta_exp_to_params(params):
    params.theta_exp = {}
    if (params.dataio.dataset_type == 'nav2d'):
        if (params.dataio.dataset_name == 'nav2dfix/dataset_0000'):
            params.theta_exp.sigma_inv_odom_vals = [1e1, 1e1, 1e1]
            params.theta_exp.sigma_inv_gps_vals = [1., 1., 1e1]
        elif (params.dataio.dataset_name == 'nav2dfix/dataset_0001'):
            params.theta_exp.sigma_inv_odom_vals = [1., 2., 1e1]
            params.theta_exp.sigma_inv_gps_vals = [2., 1., 1e1]
        elif (params.dataio.dataset_name == 'nav2dtime/dataset_0000'):
            params.theta_exp.sigma_inv_odom0_vals = [1e1, 1e1, 1e1]
            params.theta_exp.sigma_inv_gps0_vals = [1., 1., 1e1]
            params.theta_exp.sigma_inv_odom1_vals = [1., 1., 1.]
            params.theta_exp.sigma_inv_gps1_vals = [1e1, 1e1, 1e1]
        elif (params.dataio.dataset_name == 'nav2dtime/dataset_0001'):
            params.theta_exp.sigma_inv_odom0_vals = [1., 1., 1.]
            params.theta_exp.sigma_inv_gps0_vals = [1e1, 1e1, 1e1]
            params.theta_exp.sigma_inv_odom1_vals = [1e1, 1e1, 1e1]
            params.theta_exp.sigma_inv_gps1_vals = [1., 1., 1e1]

    if (params.dataio.dataset_type == 'push2d'):
        params.theta_exp.sigma_inv_tactile_rel_vals = [1, 1, 1e5]
        params.theta_exp.sigma_inv_qs_push_motion_vals = [1e3, 1e3, 1e3]

        params.theta_exp.sigma_inv_ee_pose_prior_vals = [1e4, 1e4, 1e4]
        params.theta_exp.sigma_inv_sdf_intersection_vals = [1e2]
        params.theta_exp.sigma_inv_binary_interseq_obj_vals = [1e6, 1e6, 1e6]

    return params

def init_theta(params):
    
    if (params.dataio.dataset_type == "nav2d"):
        if (params.dataio.model_type == "fixed_cov"):
            sigma_noise = params.theta_init.sigma_noise if (params.theta_init.noisy == True) else 0. 
            sigma_inv_odom_vals = np.array(params.theta_init.sigma_inv_odom_vals) + sigma_noise * np.random.randn(3)
            sigma_inv_gps_vals = np.array(params.theta_init.sigma_inv_gps_vals) + 0. * np.random.randn(3)
            theta = ThetaNav2dFixedCov(sigma_inv_odom_vals=sigma_inv_odom_vals,
                                       sigma_inv_gps_vals=sigma_inv_gps_vals)
            theta_exp = ThetaNav2dFixedCov(sigma_inv_odom_vals=params.theta_exp.sigma_inv_odom_vals,
                                           sigma_inv_gps_vals=params.theta_exp.sigma_inv_gps_vals)

        elif (params.dataio.model_type == "varying_cov"):
            sigma_noise = params.theta_init.sigma_noise if (params.theta_init.noisy == True) else 0. 
            sigma_inv_odom0_vals = np.array(params.theta_init.sigma_inv_odom0_vals) + sigma_noise * np.random.randn(3)
            sigma_inv_gps0_vals = np.array(params.theta_init.sigma_inv_gps0_vals) + sigma_noise * np.random.randn(3)
            sigma_inv_odom1_vals = np.array(params.theta_init.sigma_inv_odom1_vals) + sigma_noise * np.random.randn(3)
            sigma_inv_gps1_vals = np.array(params.theta_init.sigma_inv_gps1_vals) + sigma_noise * np.random.randn(3)
            
            theta = ThetaNav2dVaryingCov(sigma_inv_odom0_vals=sigma_inv_odom0_vals,
                                         sigma_inv_gps0_vals=sigma_inv_gps0_vals,
                                         sigma_inv_odom1_vals=sigma_inv_odom1_vals,
                                         sigma_inv_gps1_vals=sigma_inv_gps1_vals)
            theta_exp = ThetaNav2dVaryingCov(sigma_inv_odom0_vals=params.theta_exp.sigma_inv_odom0_vals,
                                             sigma_inv_gps0_vals=params.theta_exp.sigma_inv_gps0_vals,
                                             sigma_inv_odom1_vals=params.theta_exp.sigma_inv_odom1_vals,
                                             sigma_inv_gps1_vals=params.theta_exp.sigma_inv_gps1_vals)

    elif (params.dataio.dataset_type == "push2d"):
        if (params.dataio.model_type == "fixed_cov"):
            theta = ThetaPush2dFixedCov(sigma_inv_tactile_rel_vals=params.theta_init.sigma_inv_tactile_rel_vals,
                                        sigma_inv_qs_push_motion_vals=params.theta_init.sigma_inv_qs_push_motion_vals,
                                        sigma_inv_ee_pose_prior_vals=params.theta_init.sigma_inv_ee_pose_prior_vals,
                                        sigma_inv_sdf_intersection_vals=params.theta_init.sigma_inv_sdf_intersection_vals,
                                        sigma_inv_binary_interseq_obj_vals=params.theta_init.sigma_inv_binary_interseq_obj_vals)
            theta_exp = ThetaPush2dFixedCov(sigma_inv_tactile_rel_vals=params.theta_exp.sigma_inv_tactile_rel_vals,
                                            sigma_inv_qs_push_motion_vals=params.theta_exp.sigma_inv_qs_push_motion_vals,
                                            sigma_inv_ee_pose_prior_vals=params.theta_exp.sigma_inv_ee_pose_prior_vals,
                                            sigma_inv_sdf_intersection_vals=params.theta_exp.sigma_inv_sdf_intersection_vals,
                                            sigma_inv_binary_interseq_obj_vals=params.theta_exp.sigma_inv_binary_interseq_obj_vals)
        elif (params.dataio.model_type == "fixed_cov_varying_mean"):
            tactile_model_filename = "{0}/local/models/{1}.pt".format(params.BASE_PATH, params.tactile_model.name)
            theta = ThetaPush2dFixedCovVaryingMean(sigma_inv_tactile_rel_vals=params.theta_init.sigma_inv_tactile_rel_vals,
                                        sigma_inv_qs_push_motion_vals=params.theta_init.sigma_inv_qs_push_motion_vals,
                                        sigma_inv_ee_pose_prior_vals=params.theta_init.sigma_inv_ee_pose_prior_vals,
                                        sigma_inv_sdf_intersection_vals=params.theta_init.sigma_inv_sdf_intersection_vals,
                                        sigma_inv_binary_interseq_obj_vals=params.theta_init.sigma_inv_binary_interseq_obj_vals,
                                        tactile_model_filename = tactile_model_filename)
            theta_exp = ThetaPush2dFixedCovVaryingMean(sigma_inv_tactile_rel_vals=params.theta_exp.sigma_inv_tactile_rel_vals,
                                            sigma_inv_qs_push_motion_vals=params.theta_exp.sigma_inv_qs_push_motion_vals,
                                            sigma_inv_ee_pose_prior_vals=params.theta_exp.sigma_inv_ee_pose_prior_vals,
                                            sigma_inv_sdf_intersection_vals=params.theta_exp.sigma_inv_sdf_intersection_vals,
                                            sigma_inv_binary_interseq_obj_vals=params.theta_exp.sigma_inv_binary_interseq_obj_vals,
                                            tactile_model_filename = tactile_model_filename)
    else:
        print(
            "[leo_obs_models::init_theta] Observation model parameter class not found.")

    [print("[leo_obs_models::init_theta] theta: {0} {1}".format(
        name, param)) for name, param in theta.named_parameters()]
    [print("[leo_obs_models::init_theta] theta_exp: {0} {1}".format(
        name, param)) for name, param in theta_exp.named_parameters()]

    return theta, theta_exp


def min_clip(w):
    w_min = 1e-4 * torch.ones(w.shape, requires_grad=True, device=device)
    w = torch.max(w_min, w)
    return w


class ThetaNav2dFixedCov(nn.Module):
    def __init__(self, sigma_inv_odom_vals=None, sigma_inv_gps_vals=None):
        super().__init__()

        self.sigma_inv_odom = torch.nn.Parameter(torch.tensor(
            sigma_inv_odom_vals, dtype=torch.float32, device=device)) if sigma_inv_odom_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))

        self.sigma_inv_gps = torch.nn.Parameter(torch.tensor(
            sigma_inv_gps_vals, dtype=torch.float32, device=device)) if sigma_inv_gps_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))

        self.freeze_params()

    def freeze_params(self):
        pass

    def get_sigma_inv(self, factor_name, z=None):
        sigma_inv_val = getattr(self, "sigma_inv_{0}".format(factor_name))

        return sigma_inv_val

    def min_clip(self):
        self.sigma_inv_odom.data = min_clip(self.sigma_inv_odom.data)
        self.sigma_inv_gps.data = min_clip(self.sigma_inv_gps.data)

    def norm(self):
        l2_norm = torch.norm(self.sigma_inv_odom) ** 2 + torch.norm(self.sigma_inv_gps) ** 2

        return l2_norm
class ThetaNav2dVaryingCov(nn.Module):
    def __init__(self, sigma_inv_odom0_vals=None, sigma_inv_gps0_vals=None, sigma_inv_odom1_vals=None, sigma_inv_gps1_vals=None):
        super().__init__()

        self.sigma_inv_odom0 = torch.nn.Parameter(torch.tensor(
            sigma_inv_odom0_vals, dtype=torch.float32, device=device)) if sigma_inv_odom0_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))
        self.sigma_inv_gps0 = torch.nn.Parameter(torch.tensor(
            sigma_inv_gps0_vals, dtype=torch.float32, device=device)) if sigma_inv_gps0_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))

        self.sigma_inv_odom1 = torch.nn.Parameter(torch.tensor(
            sigma_inv_odom1_vals, dtype=torch.float32, device=device)) if sigma_inv_odom1_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))
        self.sigma_inv_gps1 = torch.nn.Parameter(torch.tensor(
            sigma_inv_gps1_vals, dtype=torch.float32, device=device)) if sigma_inv_gps1_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))

        self.freeze_params()

    def freeze_params(self):
        pass

    def get_sigma_inv(self, factor_name, z=None):
        sigma_inv_val0 = getattr(self, "sigma_inv_{0}0".format(factor_name))
        sigma_inv_val1 = getattr(self, "sigma_inv_{0}1".format(factor_name))

        cl = z[-1]
        sigma_inv_val = (1 - cl) * sigma_inv_val0 + cl * sigma_inv_val1

        return sigma_inv_val

    def min_clip(self):
        self.sigma_inv_odom0.data = min_clip(self.sigma_inv_odom0.data)
        self.sigma_inv_gps0.data = min_clip(self.sigma_inv_gps0.data)

        self.sigma_inv_odom1.data = min_clip(self.sigma_inv_odom1.data)
        self.sigma_inv_gps1.data = min_clip(self.sigma_inv_gps1.data)

    def norm(self):

        l2_norm = torch.norm(self.sigma_inv_odom0) ** 2 + torch.norm(self.sigma_inv_odom1) ** 2 + \
            torch.norm(self.sigma_inv_gps0) ** 2 + \
            torch.norm(self.sigma_inv_gps1) ** 2

        return l2_norm

class ThetaPush2dFixedCov(nn.Module):
    def __init__(self, sigma_inv_tactile_rel_vals=None, sigma_inv_qs_push_motion_vals=None,
                 sigma_inv_ee_pose_prior_vals=None, sigma_inv_sdf_intersection_vals=None, sigma_inv_binary_interseq_obj_vals=None):
        super().__init__()

        self.sigma_inv_tactile_rel = torch.nn.Parameter(torch.tensor(
            sigma_inv_tactile_rel_vals, device=device)) if sigma_inv_tactile_rel_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))
        self.sigma_inv_qs_push_motion = torch.nn.Parameter(torch.tensor(
            sigma_inv_qs_push_motion_vals, device=device)) if sigma_inv_qs_push_motion_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))

        self.sigma_inv_ee_pose_prior = torch.nn.Parameter(torch.tensor(
            sigma_inv_ee_pose_prior_vals, device=device)) if sigma_inv_ee_pose_prior_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))
        self.sigma_inv_sdf_intersection = torch.nn.Parameter(torch.tensor(
            sigma_inv_sdf_intersection_vals, device=device)) if sigma_inv_sdf_intersection_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))
        self.sigma_inv_binary_interseq_obj = torch.nn.Parameter(torch.tensor(
            sigma_inv_binary_interseq_obj_vals, device=device)) if sigma_inv_binary_interseq_obj_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))

        self.freeze_params()

    def freeze_params(self):
        self.sigma_inv_ee_pose_prior.requires_grad = False
        self.sigma_inv_sdf_intersection.requires_grad = False
        self.sigma_inv_binary_interseq_obj.requires_grad = False

    def get_sigma_inv(self, factor_name, z=None):
        sigma_inv_val = getattr(self, "sigma_inv_{0}".format(factor_name))

        return sigma_inv_val

    def min_clip(self):
        self.sigma_inv_tactile_rel.data = min_clip(
            self.sigma_inv_tactile_rel.data)
        self.sigma_inv_qs_push_motion.data = min_clip(
            self.sigma_inv_qs_push_motion.data)
class TactileModelNetwork(pl.LightningModule):
    def __init__(
            self, input_size, output_size=4):
        super().__init__()

        self.fc1 = torch.nn.Linear(input_size, output_size)

    def forward(self, x1, x2, k):
        x = torch.cat([x1, x2], dim=1)

        k1_ = k.unsqueeze(1)  # b x 1 x cl
        x1_ = x.unsqueeze(-1)  # b x dim x 1
        x = torch.mul(x1_, k1_)  # b x dim x cl

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return x

class ThetaPush2dFixedCovVaryingMean(nn.Module):
    def __init__(self, sigma_inv_tactile_rel_vals=None, sigma_inv_qs_push_motion_vals=None,
                 sigma_inv_ee_pose_prior_vals=None, sigma_inv_sdf_intersection_vals=None, sigma_inv_binary_interseq_obj_vals=None, tactile_model_filename=None):
        super().__init__()
        self.sigma_inv_tactile_rel = torch.nn.Parameter(torch.tensor(
            sigma_inv_tactile_rel_vals, device=device)) if sigma_inv_tactile_rel_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))
        self.sigma_inv_qs_push_motion = torch.nn.Parameter(torch.tensor(
            sigma_inv_qs_push_motion_vals, device=device)) if sigma_inv_qs_push_motion_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))

        self.sigma_inv_ee_pose_prior = torch.nn.Parameter(torch.tensor(
            sigma_inv_ee_pose_prior_vals, device=device)) if sigma_inv_ee_pose_prior_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))
        self.sigma_inv_sdf_intersection = torch.nn.Parameter(torch.tensor(
            sigma_inv_sdf_intersection_vals, device=device)) if sigma_inv_sdf_intersection_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))
        self.sigma_inv_binary_interseq_obj = torch.nn.Parameter(torch.tensor(
            sigma_inv_binary_interseq_obj_vals, device=device)) if sigma_inv_binary_interseq_obj_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))
        
        self.tactile_model = TactileModelNetwork(input_size=2*2*4, output_size=4)
        self.tactile_model = self.tactile_model.to(device)
        self.init_tactile_model_weights_file(tactile_model_filename)
        
        self.freeze_params()

    def init_tactile_model_weights_file(self, filename):

        model_saved = torch.jit.load(filename)
        state_dict_saved = model_saved.state_dict()

        # fix: saved dict key mismatch
        state_dict_new = collections.OrderedDict()
        state_dict_new['fc1.weight'] = state_dict_saved['model.fc1.weight']
        state_dict_new['fc1.bias'] = state_dict_saved['model.fc1.bias']

        self.tactile_model.load_state_dict(state_dict_new)
            
    def freeze_params(self):
        self.sigma_inv_tactile_rel.requires_grad = False
        self.sigma_inv_qs_push_motion.requires_grad = False

        self.sigma_inv_ee_pose_prior.requires_grad = False
        self.sigma_inv_sdf_intersection.requires_grad = False
        self.sigma_inv_binary_interseq_obj.requires_grad = False

    def get_sigma_inv(self, factor_name, z=None):
        sigma_inv_val = getattr(self, "sigma_inv_{0}".format(factor_name))

        return sigma_inv_val

    def tactile_model_output(self, img_feats, params=None):
        img_feat_i = torch.tensor(img_feats[0], requires_grad=True, device=device).view(1, -1)
        img_feat_j = torch.tensor(img_feats[1], requires_grad=True, device=device).view(1, -1)

        if (params.norm_img_feat == True):
            mean_img_feat = torch.tensor(params.mean_img_feat, requires_grad=True, device=device)
            std_img_feat = torch.tensor(params.std_img_feat, requires_grad=True, device=device)

            img_feat_i = tf_utils.normalize_vector(img_feat_i, mean_img_feat, std_img_feat)
            img_feat_j = tf_utils.normalize_vector(img_feat_j, mean_img_feat, std_img_feat)

        class_label = torch.tensor(params.class_label, requires_grad=False, device=device)
        class_label_vec = torch.nn.functional.one_hot(class_label, params.num_classes)
        class_label_vec = class_label_vec.view(1, -1)

        tf_pred = self.tactile_model.forward(img_feat_i, img_feat_j, class_label_vec)
        tf_pred = tf_pred.view(-1) # [tx, ty, cyaw, syaw]

        return tf_pred

    def min_clip(self):
        self.sigma_inv_tactile_rel.data = min_clip(
            self.sigma_inv_tactile_rel.data)
        self.sigma_inv_qs_push_motion.data = min_clip(
            self.sigma_inv_qs_push_motion.data)