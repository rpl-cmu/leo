from subprocess import call

import numpy as np
import math

from leopy.utils import dir_utils

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

plt.rcParams.update({'font.size': 28})
# plt.rcParams["figure.figsize"] = (12,8)

def init_plots(n_figs=1, figsize=(12, 8), interactive=True):
    if interactive:
        plt.ion()

    plt.close('all')
    figs = []
    for fid in range(0, n_figs):
        figs.append(plt.figure(constrained_layout=True, figsize=figsize))

    return figs

def write_video_ffmpeg(imgsrcdir, viddst, framerate=30, format="mp4"):

    cmd = "ffmpeg -y -r {0} -pattern_type glob -i '{1}/*.png' -c:v libx264 -pix_fmt yuv420p {2}.{3}".format(
        framerate, imgsrcdir, viddst, format)
    # cmd = "ffmpeg -y -r {0} -pattern_type glob -i '{1}/*.png' {2}.mp4".format(
    #     framerate, imgsrcdir, viddst)
    call(cmd, shell=True)

def random_color():
    color = list(np.random.choice(range(256), size=3))
    color = [x / 256 for x in color]
    return color
    
def plot_vec1d(logger, keys=None, colors=None):

    plt.cla()
    if keys is None:
        keys = [keys for keys in logger.data[0]]
    if colors is None:
        colors = [random_color() for c in range(0, len(keys))]
    
    assert(len(keys) == len(colors))

    iters = len(logger.data)
    for idx, key in enumerate(keys):
        y = [logger.data[itr][key] for itr in range(0, iters)]
        x = np.arange(0, iters)
        plt.plot(x, y, color=colors[idx], label=key, linewidth=2)

    plt.legend(loc='upper right')

    plt.show()
    plt.pause(1e-12)


def vis_step(tstep, logger, params=None):

    if (params.dataio.dataset_type == "nav2d"):
        vis_step_nav2d(tstep, logger, params)   
    elif (params.dataio.dataset_type == "push2d"):
        vis_step_push2d(tstep, logger, params)
    else:
        print("[vis_utils::vis_step] vis_step not available for {0} dataset".format(
            params.dataio.dataset_type))

def vis_step_nav2d(tstep, logger, params=None):
    
    plt.cla()
    plt.gca().axis('equal')
    plt.xlim(-70, 70)
    plt.ylim(0, 60)
    plt.axis('off')
        
    poses_graph = logger.data[tstep]["graph/poses2d"]
    poses_gt = logger.data[tstep]["gt/poses2d"]

    plt.scatter(poses_gt[tstep, 0], poses_gt[tstep, 1], marker=(3, 0, poses_gt[tstep, 2]/np.pi*180),
                color='dimgray', s=800, alpha=0.5, zorder=3)
    
    plt.plot(poses_gt[0:tstep, 0], poses_gt[0:tstep, 1], linewidth=4, linestyle='--',
                color=params.plot.colors.gt, label=params.plot.labels.gt)
    plt.plot(poses_graph[0:tstep, 0], poses_graph[0:tstep, 1], linewidth=3,
                color=params.plot.colors.opt, label=params.plot.labels.opt, alpha=0.8)

    # plt.legend(loc='upper right')

    if params.optim.show_fig:
        plt.show()
        plt.pause(1e-12)
    
    if params.optim.save_fig:
        filename = "{0}/{1}/{2}/{3}/{4}/{5:04d}.png".format(
            params.BASE_PATH, params.plot.dstdir, params.dataio.dataset_name, params.dataio.prefix, params.dataio.data_idx, params.leo.itr)
        # print("[vis_utils::vis_step_nav2d] Save figure to {0}".format(filename))
        dir_utils.save_plt_fig(plt, filename, mkdir=True)

def draw_endeff(poses_ee, color="dimgray", label=None, ax=None):

    # plot contact point and normal
    plt.plot(poses_ee[-1][0], poses_ee[-1][1],
             'k*') if ax is None else ax.plot(poses_ee[-1][0], poses_ee[-1][1], 'k*')
    ori = poses_ee[-1][2]
    sz_arw = 0.03
    (dx, dy) = (sz_arw * -math.sin(ori), sz_arw * math.cos(ori))
    plt.arrow(poses_ee[-1][0], poses_ee[-1][1], dx, dy, linewidth=2,
              head_width=0.001, color=color, head_length=0.01, fc='dimgray', ec='dimgray') if ax is None else ax.arrow(poses_ee[-1][0], poses_ee[-1][1], dx, dy, linewidth=2,
                                                                                                                       head_width=0.001, color=color, head_length=0.01, fc='dimgray', ec='dimgray')

    ee_radius = 0.0075
    circle = mpatches.Circle(
        (poses_ee[-1][0], poses_ee[-1][1]), color='dimgray', radius=ee_radius)
    plt.gca().add_patch(circle) if ax is None else ax.add_patch(circle)


def draw_object(poses_obj, shape="disc", color="dimgray", label=None, ax=None):

    linestyle_gt = '--' if (color == "dimgray") else '-'
    plt.plot(poses_obj[:, 0], poses_obj[:, 1], color=color,
             linestyle=linestyle_gt, label=label, linewidth=2, alpha=0.9) if ax is None else ax.plot(poses_obj[:, 0], poses_obj[:, 1], color=color,
                                                                                                     linestyle=linestyle_gt, label=label, linewidth=2, alpha=0.9)

    if (shape == "disc"):
        disc_radius = 0.088
        circ_obj = mpatches.Circle((poses_obj[-1][0], poses_obj[-1][1]), disc_radius,
                                   facecolor='None', edgecolor=color, linestyle=linestyle_gt, linewidth=2, alpha=0.9)
        plt.gca().add_patch(circ_obj) if ax is None else ax.add_patch(circ_obj)

        # cross-bars
        (x0, y0, yaw) = (poses_obj[-1][0],
                         poses_obj[-1][1], poses_obj[-1][2])
        r = disc_radius
        plt.plot([x0 + r * math.cos(yaw), x0 - r * math.cos(yaw)],
                 [y0 + r * math.sin(yaw), y0 - r * math.sin(yaw)],
                 linestyle=linestyle_gt, color=color, alpha=0.4) if ax is None else ax.plot([x0 + r * math.cos(yaw), x0 - r * math.cos(yaw)],
                                                                                            [y0 + r * math.sin(
                                                                                                yaw), y0 - r * math.sin(yaw)],
                                                                                            linestyle=linestyle_gt, color=color, alpha=0.4)
        plt.plot([x0 - r * math.sin(yaw), x0 + r * math.sin(yaw)],
                 [y0 + r * math.cos(yaw), y0 - r * math.cos(yaw)],
                 linestyle=linestyle_gt, color=color, alpha=0.4) if ax is None else ax.plot([x0 - r * math.sin(yaw), x0 + r * math.sin(yaw)],
                                                                                            [y0 + r * math.cos(
                                                                                                yaw), y0 - r * math.cos(yaw)],
                                                                                            linestyle=linestyle_gt, color=color, alpha=0.4)

    elif (shape == "rect"):
        rect_len_x = 0.2363
        rect_len_y = 0.1579

        yaw = poses_obj[-1][2]
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw), np.cos(yaw)]])
        offset = np.matmul(R, np.array(
            [[0.5*rect_len_x], [0.5*rect_len_y]]))
        xb = poses_obj[-1][0] - offset[0]
        yb = poses_obj[-1][1] - offset[1]
        rect = mpatches.Rectangle((xb, yb), rect_len_x, rect_len_y, angle=(
            np.rad2deg(yaw)), facecolor='None', edgecolor=color, linestyle=linestyle_gt, linewidth=2)
        plt.gca().add_patch(rect) if ax is None else ax.add_patch(rect)

    elif (shape == "ellip"):
        ellip_len_x = 0.1638
        ellip_len_y = 0.2428

        xb = poses_obj[-1][0]
        yb = poses_obj[-1][1]
        yaw = poses_obj[-1][2]
        ellip = mpatches.Ellipse((xb, yb), ellip_len_x, ellip_len_y, angle=(
            np.rad2deg(yaw)), facecolor='None', edgecolor=color, linestyle=linestyle_gt, linewidth=2)
        plt.gca().add_patch(ellip) if ax is None else ax.add_patch(ellip)

def vis_step_push2d(tstep, logger, params=None):
    
    plt.cla()
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.gca().axis('equal')
    plt.axis('off')

    poses_obj_gt = np.array(logger.data[tstep]["gt/obj_poses2d"])
    poses_ee_gt = np.array(logger.data[tstep]["gt/ee_poses2d"])
    draw_endeff(poses_ee_gt, color="dimgray")
    draw_object(poses_obj_gt, shape=params.dataio.obj_shape,
                color="dimgray", label="groundtruth")
    
    color = params.plot.colors.exp if (params.optim.fig_name == "expert") else params.plot.colors.opt
    label = params.plot.labels.exp if (params.optim.fig_name == "expert") else params.plot.labels.opt
    poses_obj_graph = np.array(logger.data[tstep]["graph/obj_poses2d"])
    poses_ee_graph = np.array(logger.data[tstep]["graph/ee_poses2d"])
    draw_endeff(poses_ee_graph, color=color)
    draw_object(poses_obj_graph, shape=params.dataio.obj_shape,
                color=color, label=label)
    # plt.legend(loc='upper left')

    if params.optim.show_fig:
        plt.draw()
        plt.pause(1e-12)
    
    if params.optim.save_fig:
        filename = "{0}/{1}/{2}/{3}/{4}/{5:04d}.png".format(
            params.BASE_PATH, params.plot.dstdir, params.dataio.dataset_name, params.dataio.prefix, params.dataio.data_idx, params.leo.itr)
        # print(f"[vis_utils::vis_step_push2d] Save figure to {filename}")
        dir_utils.save_plt_fig(plt, filename, mkdir=True)