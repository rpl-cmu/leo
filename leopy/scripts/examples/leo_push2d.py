#!/usr/bin/env python

import os
import hydra

from leopy.algo import leo_update

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH = os.path.join(BASE_PATH, "config/examples/leo_push2d.yaml")

@hydra.main(config_name=CONFIG_PATH)
def main(cfg):

    cfg.BASE_PATH = BASE_PATH
    leo_update.run(cfg)

if __name__ == '__main__':
    main()