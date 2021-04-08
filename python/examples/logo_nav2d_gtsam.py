
#!/usr/bin/env python

import os
import json
import hydra

from logopy.algo import logo_update

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH = os.path.join(BASE_PATH, "python/config/logo_nav2d_gtsam.yaml")

@hydra.main(config_name=CONFIG_PATH, strict=False)
def main(cfg):

    print(cfg.pretty())

    cfg.BASE_PATH = BASE_PATH
    logo_update.run(cfg)

if __name__ == '__main__':
    main()