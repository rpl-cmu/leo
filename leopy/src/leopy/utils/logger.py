import numpy as np
from collections import defaultdict
import pandas as pd

import logging
log = logging.getLogger(__name__)

class Logger():

    def __init__(self, params=None):
        self.params = params
        self.data = defaultdict(dict)
        self.runtime = None

    def data(self):
        return self.data

    def params(self):
        return self.params

    def runtime(self):
        return self.runtime
    
    def log_step(self, name, value, step):
        self.data[step][name] = value
    
    def log_param(self, name, value):
        self.params[name] = value
    
    def log_runtime(self, runtime):
        self.runtime = runtime