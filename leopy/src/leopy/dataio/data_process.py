import sys
sys.path.append("/usr/local/cython/")

import leopy.utils.tf_utils as tf_utils
import gtsam
import math
import numpy as np

def transform_episodes_common_frame(episode_list, push_data):

    push_data_tf = push_data

    push_data_idxs = []
    episode_idxs = []
    for episode in episode_list:
        push_data_idxs_curr = [idx for idx, val in enumerate(
            push_data['contact_episode']) if (val == episode)]
        push_data_idxs.append(push_data_idxs_curr)
        episode_idxs.append([episode] * len(push_data_idxs_curr))

    push_data_idxs = [item for sublist in push_data_idxs for item in sublist]
    episode_idxs = [item for sublist in episode_idxs for item in sublist]
    num_steps = len(push_data_idxs)

    pose_idx, prev_pose_idx = -1, -1

    obj_prevseq_last_pose = None
    prevseq_pose_set = False
    for tstep in range(num_steps):

        prev_pose_idx = pose_idx
        pose_idx = pose_idx + 1
        push_idx = push_data_idxs[pose_idx]

        if ((tstep != 0) & (episode_idxs[prev_pose_idx] != episode_idxs[pose_idx])):

            prev_push_idx = push_data_idxs[prev_pose_idx]
            obj_prevseq_last_pose = tf_utils.vec3_to_pose2(push_data['obj_poses_2d'][prev_push_idx])
            obj_newseq_first_pose = tf_utils.vec3_to_pose2(push_data['obj_poses_2d'][push_idx])
        
            prevseq_pose_set = True

        if prevseq_pose_set:
            obj_pose = tf_utils.vec3_to_pose2(push_data['obj_poses_2d'][push_idx])            
            ee_pose = tf_utils.vec3_to_pose2(push_data['ee_poses_2d'][push_idx])

            # transform obj pose
            obj_pose_tf = obj_prevseq_last_pose.compose(obj_newseq_first_pose.between(obj_pose))

            # transform endeff pose
            ee_pose_tf = obj_pose_tf.compose(obj_pose.between(ee_pose))

            push_data_tf['obj_poses_2d'][push_idx] = tf_utils.pose2_to_vec3(obj_pose_tf)
            push_data_tf['ee_poses_2d'][push_idx] = tf_utils.pose2_to_vec3(ee_pose_tf)

    return push_data_tf
