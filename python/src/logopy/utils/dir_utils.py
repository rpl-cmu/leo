import os
import pickle as pkl

import csv
import json

def make_dir(dir, verbose=False, clear=False):
    if verbose:
        print("Creating directory {0}".format(dir))

    cmd = "mkdir -p {0}".format(dir)
    os.popen(cmd, 'r')

    if clear:
        cmd = "rm -rf {0}/*".format(dir)
        os.popen(cmd, 'r')

def load_pkl_obj(filename):
    with (open(filename, "rb")) as f:
        pkl_obj = pkl.load(f)
    f.close()

    return pkl_obj

def save_pkl_obj(filename, pkl_obj):
    f = open(filename, 'wb')
    pkl.dump(pkl_obj, f)
    f.close()

def write_file_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print("Saved data to file: {}".format(filename))

def read_file_json(filename, verbose=False):
    data = None
    with open(filename) as f:
        data = json.load(f)

    if verbose:
        print("Loaded file: {0}". format(filename))

    return data

def write_file_csv(filename, data, fieldnames):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(data)
    print("Saved data to file: {}".format(filename))

def write_dict_of_lists_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))

def save_plt_fig(plt, filename, mkdir=True):
    if mkdir:
        dstdir = "/".join(filename.split("/")[:-1])
        if not os.path.exists(dstdir):
            make_dir(dstdir)

    plt.savefig(filename)
